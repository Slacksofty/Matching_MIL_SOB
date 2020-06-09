import pandas as pd
import numpy as np
import time, re, math, sys, getopt, pyodbc, unidecode, urllib
from scipy import sparse
from pathlib2 import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine

#Variables globales
mil_query = '''SELECT CD_PRODUIT AS CODE_MIL, LB_PRODUIT AS LABEL_MIL, LIBELLE_APPELLATION AS APPELLATION_MIL 
                                FROM MILLESIMA_PRODUITS
                                INNER JOIN MILLESIMA_APPELLATIONS_NEW
                                ON MILLESIMA_PRODUITS.CD_APPELLATION_NEW = MILLESIMA_APPELLATIONS_NEW.ID_APPELLATION'''

sob_query = '''SELECT CD_SOUS_GROUPE AS CODE_SOB, LB_SOUS_GROUPE , SOBOVI_APPELLATIONS.LB_GROUPE AS APPELLATION_SOB, SOBOVI_PRODUITS.CD_COULEUR, CONCAT(LB_SOUS_GROUPE, ' ', SOBOVI_PRODUITS.CD_COULEUR) AS LABEL_SOB
                                FROM SOBOVI_PRODUITS
                                INNER JOIN SOBOVI_APPELLATIONS
                                ON SOBOVI_PRODUITS.CD_GROUPE = SOBOVI_APPELLATIONS.CD_GROUPE'''

mil_columns_names = ['CODE_MIL', 'LABEL_MIL', 'APPELLATION_MIL']
sob_columns_names = ['CODE_SOB', 'LABEL_SOB', 'APPELLATION_SOB']

server_name = 'srv-v-bdd01'
DB_name = 'TF-IDF'
user_name = 'Test_TFIDF'
password = 'TFIDF_Test'

# Pull une DataFrame depuis la BD
def setup_DataFrame(sql_query, columns_names, conn):
	df_query = pd.read_sql_query(sql_query, conn)
	df = pd.DataFrame(df_query, columns=columns_names)
	df[df.columns[1]] = [" ".join(re.sub('[^A-Za-z0-9 ]+', '', ((unidecode.unidecode(s).replace('(r)', 'rouge ')).replace('(b)', 'blanc '))).split()).upper() for s in df[df.columns[1]]]
	df[df.columns[2]] = [" ".join(re.sub('[^A-Za-z0-9 ]+', '', unidecode.unidecode(s)).split()).upper() for s in df[df.columns[2]]]
	return df

# Fonction de decoupage des mots avec n la longueur des mots en nombre de characteres
def ngrams(string, n=3):
	ngrams = zip(*[string[i:] for i in range(n)])
	return [''.join(ngram) for ngram in ngrams]


# Fonction de creation des lignes pour la table de matching des labels
# Retourne un dictionnaire
def create_df_lbl(m, row_indexes, col_indexes, df_a, df_b):
	buffer = []
	for i in range(m.data.size):
		if(df_a[df_a.columns[3]][row_indexes[i]] == df_b[df_b.columns[3]][col_indexes[i]]):
				buffer.append(dict(CODE_MIL=df_a[df_a.columns[0]][row_indexes[i]], 
				CODE_SOB=df_b[df_b.columns[0]][col_indexes[i]], 
				LABEL_MIL=df_a[df_a.columns[1]][row_indexes[i]], 
				LABEL_SOB=df_b[df_b.columns[1]][col_indexes[i]], 
				SIMILARITY=math.sqrt(m.data[i])))
	return buffer

# Fonction de creation des lignes pour la table de matching des appellations
# Retourne un dictionnaire
def create_df_apl(m, row_indexes, col_indexes, df_a, df_b):
	buffer = []
	for i in range(m.data.size):
		buffer.append(dict(APL_MIL=df_a[row_indexes[i]], 
		APL_SOB=df_b[col_indexes[i]], 
		SIMILARITY=m.data[i]))
	return buffer

# Creer une matrice TF-IDF pour chaque corpus a et b
# Effectue une conversion en CSR
# Retourne sous forme de matrice dense la similarite cosine entre les 2 matrices TF-IDF 
def matching(a,b):
	vectorizer1 = TfidfVectorizer(min_df=1, analyzer=ngrams)
	X = sparse.csr_matrix(vectorizer1.fit_transform(a))
	vectorizer2 = TfidfVectorizer(vocabulary=vectorizer1.get_feature_names(), min_df=1, analyzer=ngrams)
	Y = sparse.csr_matrix(vectorizer2.fit_transform(b))
	return cosine_similarity(X,Y, dense_output=True)

# Extrait les matchs a partir d'une matrice de similarite cosine
# Peut accepter une deuxieme matrice m2 et procde alors a une multiplication terme a terme
# Le but est de renforcer la qualite des matchs en effectuant un matching inverse
# 'create_df_func' specifie la fonction de creation de dict a utiliser
# 'threshold' est le seuil de similarite acceptable
# Retourne une DataFrame
def extraction(m1, df_a, df_b, create_df_func, threshold, verbose, m2=None):
	if(m2 is not None):
		m1[m1 < threshold] = 0
		m2[m2 < threshold] = 0
		m = np.multiply(m1, m2)
		m[m < threshold**2] = 0
	else: 
		m = m1
		m[m < threshold] = 0
	m = sparse.csr_matrix(m)
	row_indexes = m.nonzero()[0]
	col_indexes = m.nonzero()[1]
	df = pd.DataFrame(create_df_func(m, row_indexes, col_indexes, df_a, df_b))
	if(df.columns.size > 0): df = df.sort_values(by=df.columns[df.columns.size-1],ascending=False)
	for i in range(df.columns.size-1):
		df = df.drop_duplicates(subset=df.columns[i])
	if(verbose): print(df)
	return df

# Extrait une liste d'appelations a partir d'une DataFrame
def apl_setup(df):
	return list(dict.fromkeys([" ".join(re.sub('[^A-Za-z0-9 ]+', '', lb).split()).upper() for lb in df[df.columns[2]]]))
	

# Creer une liste de numero pour chaque label de millesima et sobovi
# Ce numero correspond a l'index de l'appellation du label dans la table des appellations matchees
# Si le label n'a pas d'appellation ou une appellation qui n'a pas de match aucun numero n'est ajoute
# Ajoute une colonne correspondant a l'index a la DataFrame df
def apl_indexing(df, i, matches_apl):
	apl_index= []
	for lb in df[df.columns[2]]:
		if not(matches_apl.loc[matches_apl[matches_apl.columns[i]] == lb].empty):
			apl_index.append(matches_apl.loc[matches_apl[matches_apl.columns[i]] == lb][matches_apl.columns[0]].tolist()[0])
		else: apl_index.append('')
	df['INDEX_APPELLATION'] = apl_index
	

def main(argv):

	#Debut de mesure du temps d'execution
	t_start = time.time()

	#Options par default 
	threshold_apl = 0.6
	threshold_lbl = 0.6
	output = 'output'
	verbose = False

	#Traitement des options
	short_options = 'hvo:a:l:'
	long_options = ['help', 'verbose', 'output=', 'apl=', 'lbl=']

	try:
		args, _values = getopt.getopt(argv, short_options, long_options)
	except getopt.error as err:
		print(str(err) + '\nUtilisez -h ou --help pour afficher l\'aide')
		sys.exit(2)

	for arg, val in args:
		if arg in ('-h', '--help'):
			print('Usage : matching.py -o <outputFile> -a <thresholdApl(default = 0.6)> -l <thresholdLbl(default = 0.6)> -v : verbose')
		elif arg in ('-v', '--verbose'):
			verbose = True
		elif arg in ('-o', '--output'):
			output = val
		elif arg in ('-a', '--apl'):
			threshold_apl = float(val)
		elif arg in ('-l', '--lbl'):
			threshold_lbl = float(val)
    	
	#Connection a la BD
	params = urllib.parse.quote_plus('Driver={SQL Server};' + f'Server={server_name}; Database={DB_name}; Trusted_Connection=no; UID={user_name}; PWD={password};')
	engine = create_engine(f'mssql+pyodbc:///?odbc_connect={params}')

	mil = setup_DataFrame(mil_query, mil_columns_names, engine)
	sob = setup_DataFrame(sob_query, sob_columns_names, engine)

	apl_mil = apl_setup(mil)
	apl_sob = apl_setup(sob)

	tfidf_matrix_apl = matching(apl_mil, apl_sob)

	matches_apl = extraction(tfidf_matrix_apl, apl_mil, apl_sob, create_df_apl, threshold_apl, verbose)

	apl_indexing(mil, 0, matches_apl)
	apl_indexing(sob, 1, matches_apl)

	tfidf_matrix_1 = matching(mil[mil.columns[1]], sob[sob.columns[1]])
	tfidf_matrix_2 = matching(sob[sob.columns[1]], mil[mil.columns[1]]).transpose()

	matches = extraction(tfidf_matrix_1, mil, sob, create_df_lbl, threshold_lbl, verbose, m2 = tfidf_matrix_2)
	matches.to_csv(Path(f'../res/{output}.csv'), index=False)
	matches.to_sql('MATCHING_RESULTS', con=engine, if_exists='replace', index=False)

	#affichage du temps d'execution
	t_end = time.time() - t_start
	print(t_end)

main(sys.argv[1:])
