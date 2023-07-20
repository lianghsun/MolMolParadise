import streamlit as st
import pandas as pd
import swifter
import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, PandasTools
from rdkit.Chem.Draw import rdMolDraw2D, MolDraw2DSVG
from rdkit.ML.Cluster import Butina
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric
import numpy as np
import pandas as pd
from tqdm import tqdm
import subprocess
import time

import bokeh
import umap
# from bokeh.plotting import figure, show, output_notebook
from bokeh.plotting import figure, show, output_file, save
from bokeh.palettes import viridis
from bokeh.io import output_notebook

from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral4

st.set_page_config(page_title='MolMolParadise', page_icon='🗺️', layout="wide")
st.title('🗺️ MolMolParadise')
st.markdown('_Getting your feature vectors is as simple as eating **shabu-shabu**._ 🍲')

## Session state
# if ''

## Helper
@st.cache_data
def load_data(path):
    frame = PandasTools.LoadSDF(path, smilesName='smiles')
    # frame.drop(columns=['ROMol'], inplace=True)
    return frame

@st.cache_data
def load_upload(file, ext, **kwargs):
    smiles_col_name = kwargs['smilesName']
    if ext == 'text/csv':
        frame = pd.read_csv(uploaded_file)
        PandasTools.AddMoleculeColumnToFrame(frame, smiles_col_name, includeFingerprints=True)
        return frame
    elif ext == 'application/octet-stream':
        return PandasTools.LoadSDF(uploaded_file, smilesName=smiles_col_name)
    else:
        raise "Errors"

def GetCentroid(frame, cutoff=0.3):
    fps = [GetMorganFingerprintAsBitVect(m,2) for m in frame['ROMol']]
    distmat = []
    dists = []
    nfps = len(fps)
    
    if nfps == 1: return frame
    for idx in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], fps[:idx])
        dists.extend([1 - sim for sim in sims])

    cs = Butina.ClusterData(dists, nfps, cutoff, isDistData=True)
    return frame.iloc[list(cs[0])] # The first element for each cluster is its centroid.

def GetGenericMurckoSmiles(m):
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m)
    return Chem.MolToSmiles(MakeScaffoldGeneric(GetScaffoldForMol(m)))

def mol2svg(mol):
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def smi2svg(smi):
    mol = Chem.MolFromSmiles(smi)
    d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()

def GetFingerprint(m):
    if isinstance(m, str):
        m = Chem.MolFromSmiles(m)
    return GetMorganFingerprintAsBitVect(m, 2)

def GenericMurckoCentroids(frame):
    frame['murcko_generic'] = frame['ROMol'].apply(GetGenericMurckoSmiles)
    frame['fingerprint'] = frame['ROMol'].apply(GetFingerprint)
    frame['murcko_fingerprint'] = frame['murcko_generic'].apply(GetFingerprint)
    
    cenlist = [GetCentroid(grp, cutoff=0.1) for murcko, grp in tqdm(frame.groupby('murcko_generic'))]
    return pd.concat(cenlist)

def GetHits(id, hits):
    return True if id in hits else False

def download_file():
    with open('test.csv', 'rb') as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="test.csv">Download CSV File</a>'
        return href
    
# @st.experimental_memo
# def GetMorganFP(frame):
#     return GetMorganFingerprintAsBitVect(frame['ROMol'], 3)

def GetMurckoMorganFP(frame):
    murcko_mol = MakeScaffoldGeneric(GetScaffoldForMol(frame['ROMol']))
    return GetMorganFingerprintAsBitVect(murcko_mol, 3)

# TODO: fix
def GetImageMolFP(frame):
    command = "docker run -it --gpus 'device=6' --shm-size=8gb -v /home/owen/Code/MolMolParadise:/data imagemol --input /data/smiles.csv"
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
#     imagemol_arr = np.load('./ImageMol/ImageMol_dict.npy',  allow_pickle=True)
#     imagemol_list = imagemol_arr.tolist()
#     # st.write(imagemol_list.values())
#     _ = [{'imagemol_features': np.array(value)} for key, value in imagemol_list.items()]
#     imagemol_df = pd.DataFrame(_)
#     st.write(imagemol_df)
    
#     # frame = frame.merge(imagemol_df, how='right', on='SMILES')
#     # PandasTools.AddMoleculeColumnToFrame(frame, 'SMILES', 'ROMol', includeFingerprints=True)
#     # frame.reset_index(drop=True)
#     frame = pd.concat([frame, imagemol_df], axis=1)
#     # st.write(pd.Series(imagemol_list))
#     # frame['imagemol_features'] = pd.Series(imagemol_list.values())
#     st.write(frame['imagemol_features'])
    
    imagemol_arr = np.load('./ImageMol/ImageMol_dict.npy',  allow_pickle=True)
    imagemol_list = imagemol_arr.tolist()
    _ = []
    for idx, (key, value) in enumerate(imagemol_list.items()):
        _.append({'SMILES': key, 'imagemol_features': np.array(value, dtype='f')})
    imagemol_df = pd.DataFrame(_)
    frame = pd.merge(frame, imagemol_df)
    # st.write(type(frame['imagemol_features'][0]))
    
    return frame

# @st.cache_data
def GetProjection(frame, fp_col):
    features = np.array(frame[fp_col].values.tolist()).squeeze() 
    
    reducer = umap.UMAP(random_state=42, n_jobs=32)
    embed = reducer.fit_transform(features)
    embed_df = pd.DataFrame(embed, columns=[f'{fp_col}_x', f'{fp_col}_y'])
    
    point_df = pd.concat([frame, embed_df], axis=1)
    point_df.reset_index(drop=True)
    return point_df

def DrawUMAP(frame, x, y, selected={}, title=''):
    fig = figure(title=f'UMAP of {title}')
    
    frame = frame[['id', 'SMILES', 'image', x, y]]
    source = ColumnDataSource(frame)
    fig.circle(
            x=x, y=y,
            source=source,
            color='#cccccc',
            legend_label=f'All molecules ({frame.__len__():,})'
    )
    
    ###
    # selected = {
    #    'class_1': ['id-01', 'id-02'],
    #    'class_2': ['id-03', 'id-04']
    # }
    ###
    selected = {key: value for key, value in selected.items() if value}
    if len(selected):
        colors = viridis(len(selected))
        for color, (key, value) in zip(colors, selected.items()):
            frame[key] = False
            frame.loc[frame['id'].isin(value), key] = True
            fig.circle(
                x=x, y=y,
                source=ColumnDataSource(frame[frame[key] == True]),
                color=color,
                legend_label=f"{key} ({frame[frame[key] == True].__len__():,})"
            )
            
    TOOLTIPS = [
        ("ID", "@id"),
        ("Mol", "@image{safe}"),
    ]
    fig.add_tools(HoverTool(tooltips=TOOLTIPS))
    fig.legend.click_policy="hide"
    st.bokeh_chart(fig, use_container_width=False)
    
##

st.header('Step 1: Choose your library')
vs_src = st.radio(
    "Upload your own library or use pre-define one.",
    ('📋 Pre-define', '📤 Upload'),
    horizontal=True
)
if vs_src == '📋 Pre-define':
    selected_vs = st.selectbox(
        'Currently we provide limited libraries for testing.', 
        [
            '[LifeChemical] Helicase Focused Library',
            # '[Asinex] Nucleosied Mimetics Library',
            # '[Asinex] Dinucleoside Mimetics for RNA Library',
            # '[ChemDiv] Nucleoside Mimetics Library'
        ]

    )
    
    vs_path = {
        '[LifeChemical] Helicase Focused Library': '/home/owen/Code/sanofi-dhx33/datasets/LC_Helicase_Focused_Library.sdf',
        # '[LifeChemical] Helicase Focused Library': '/home/owen/Code/Test/test.sdf',
        # '[Asinex] Nucleosied Mimetics Library': '',
        # '[Asinex] Dinucleoside Mimetics for RNA Library': '',
        # '[ChemDiv] Nucleoside Mimetics Library': ''

    }
    vs = vs_path[selected_vs]
    frame = load_data(vs)
    
elif vs_src == '📤 Upload':
    st.info('ℹ️ When using an uploaded file, it is necessary to specify the names of the **`SMILES`** and **`ID`** columns.')
    
    col1, col2 = st.columns(2)
    with col1:
        if 'smiles_col_name' not in st.session_state:
            st.session_state.smiles_col_name = 'smiles'
        st.session_state.smiles_col_name = st.text_input('`SMILES` Column Name', value='smiles', placeholder='Please specify the name of SMILES column')
    with col2:
        if 'id_col_name' not in st.session_state:
            st.session_state.id_col_name = 'IDNUMBER'
        st.session_state.id_col_name = st.text_input('`ID` Column Name', value='IDNUMBER', placeholder='Please specify the name of ID column')
    
    if st.session_state.smiles_col_name and st.session_state.id_col_name:
        uploaded_file = st.file_uploader(
            "Choose a file in `sdf` or `csv` format",
            type=['csv', 'sdf']
        )
        if uploaded_file is not None:
            if uploaded_file.type == "text/csv":
                # frame = pd.read_csv(uploaded_file)
                # PandasTools.AddMoleculeColumnToFrame(frame, smiles_col_name, includeFingerprints=True)
                frame = load_upload(uploaded_file, 'text/csv', smilesName=st.session_state.smiles_col_name)

            elif uploaded_file.type == "application/octet-stream":
                # frame = PandasTools.LoadSDF(uploaded_file, smilesName=smiles_col_name)
                frame = load_upload(uploaded_file, 'application/octet-stream', smilesName=st.session_state.smiles_col_name)
    else:
        uploaded_file = st.file_uploader(
            "Choose a file in `sdf` or `csv` format",
            type=['csv', 'sdf'],
            disabled=True
        )

# st.write(vs)
if 'frame' in locals() or 'frame' in globals():
    frame.rename({'IDNUMBER': 'id', 'smiles': 'SMILES'}, axis='columns', inplace=True)
    # frame = frame.sample(10)
    st.write(frame.head(5))
    st.markdown(f'_Note: The total is `{frame.__len__():,}`, but only the first **5** entries are listed._')


st.header('Step 2: Pick up feature types')

if 'feature_type' not in st.session_state:
    st.session_state.feature_type = [
        'Murcko Morgan FP',
        'ImageMol'
    ]

st.session_state.feature_type = st.multiselect(
    label='Select your desired feature types.',
    options=[
        'SMILES Morgan FP',
        'Murcko Morgan FP',
        'ImageMol'
    ],
    default=[
        # 'SMILES Morgan FP',
        'Murcko Morgan FP',
        'ImageMol'
    ]
)
    
col1, col2 = st.columns(2)
with col1:
    with st.expander('⚙️ __UMAP__ setting _(Coming Soon)_'):
        if 'projection' not in st.session_state:
            st.session_state.projection = True
            
        st.session_state.projection = st.checkbox(
            label='Project to __UMAP__',
            value=True,
            help="We use the UMAP algorithm to compute a 2D projection with default parameters. Please note that this step can be time-consuming when dealing with a large amount of data."
        )
        if st.session_state.projection:
            if 'umap_' not in st.session_state:
                st.session_state.umap_ = True 
            st.session_state.umap_ = st.checkbox(
                label='Display __UMAP projection__ visualization',
                value=True,
                help=""
        )

        else:
            if 'umap_' not in st.session_state:
                st.session_state.umap_ = False 
            st.session_state.umap_ = st.checkbox(
                label='Display __UMAP projection__ visualization',
                value=False,
                help="",
                disabled=True
            )
with col2:
    with st.expander('⚙️ __Cluster__ setting _(Coming Soon)_'):
        if all([st.session_state.umap_, 'ImageMol' in st.session_state.feature_type]):
            if 'is_cluster' not in st.session_state:
                st.session_state.is_cluster = True             
            st.session_state.is_cluster = st.checkbox(
                label='Use clustering',
                value=True,
                help=""
        )
        else:
            if 'is_cluster' not in st.session_state:
                st.session_state.is_cluster = False   
            st.session_state.is_cluster = st.checkbox(
                label='Use clustering',
                value=False,
                help=""
        )
            
        if st.session_state.is_cluster:
            if 'cluster_method' not in st.session_state:
                st.session_state.cluster_method = False
            st.session_state.cluster_method = st.selectbox(
                label='🔘 How to cluster?',
                options=[
                    'K-Means',
                    'HDBSCAN'
                ],
            )
        else:
            st.session_state.cluster_method = st.selectbox(
                label='🔘 How to cluster?',
                options=[
                    'K-Means',
                    'HDBSCAN'
                ],
                disabled=True
            )
            
        if 'K-Means' in st.session_state.cluster_method:
            k = st.number_input(
                label='🔢 __K__ value',
                value=10,
                step=1,
                min_value=2,
                max_value=20,
                help="",
                disabled=False
            )
        
st.header('(Optional) Step 3: Highlight molecules')
is_highlight = st.checkbox(
    label='🔦 Use __highlight__',
    value=False,
    help=""
)
if 'class_name' not in st.session_state:
    st.session_state.class_name = []
if 'select_idx' not in st.session_state:
    st.session_state.select_idx = []
    
if is_highlight:
    # inputs = st.text_input('_(Optional)_ __🔦 Highlight selected molecules__: for multiple IDs, please use comma to seperate each of them. Example ids: `F6541-5830, F6568-0745`', placeholder='F6541-5830, F6568-0745')
    # if inputs:
    #     id_list = inputs.split(',')
    #     id_list = [idx.strip() for idx in id_list]
    #     selected = {f'class_1': id_list}
    # else:
    #     selected = {}

    col1, col2 = st.columns([0.2, 0.8])

    with col1:
        class_num = st.number_input(
            label='🗂️ How many class?',
            value=0,
            step=1,
            min_value=0,
            max_value=20,
            help="",
            disabled=False
        )
    with col2:
        st.write('')

    class_name = []
    select_idx = []
    for idx in range(class_num):
        col1, col2 = st.columns([0.2, 0.8])
        with col1:
            st.session_state.class_name.append(st.text_input(f'__🔖 Class {idx+1}__ name', value=f'Class {idx+1}', placeholder='Give some coooool name of this class'))
        with col2:
            st.session_state.select_idx.append(st.multiselect('__🔦 Highlight selected molecules__', options=frame.id.to_list(), key=f"select-input-{idx+1}"))
        
if all(
    [any(['frame' in locals(), 'frame' in globals()]), 
     len(st.session_state.feature_type) > 0]
):
    if st.button('🥢 _Ready for **shabu-shabu**_ !'):
        with st.spinner('⌛ Wait for it...'):
            # SMILES FP section
            ## Get Fingerprint
            
            if 'SMILES Morgan FP' in st.session_state.feature_type:
                with st.spinner('⌛ Processing __SMILES FP__...'): 
                    start = time.time()
                    # frame['fingerprint'] = frame['SMILES'].swifter.apply(lambda smi: Chem.MolFromSmiles(smi)).apply(lambda m: GetMorganFingerprintAsBitVect(m, 3))
                    frame['fingerprint'] = frame['ROMol'].swifter.apply(lambda m: GetMorganFingerprintAsBitVect(m, 3))
                    
                    elapsed = time.time() - start

                st.info(f"__Fingerprint__ `{round(elapsed,2)}`s left...")

                ## Get Projection
                if st.session_state.projection:
                    with st.spinner('⌛ Processing __SMILES FP__ Projection...'): 
                        start = time.time()
                        frame = GetProjection(frame, 'fingerprint')
                        elapsed = time.time() - start
                    st.info(f"__Fingerprint Projection__ `{round(elapsed,2)}`s left...")
            
            # Murcko FP section
            ## Get Finerprint
            if 'Murcko Morgan FP' in st.session_state.feature_type:
                with st.spinner('⌛ Processing __Murcko SMILES FP__...'):
                    start = time.time()
                    # frame['murcko_fingerprint'] = frame['SMILES'].swifter.apply(lambda smi: Chem.MolFromSmiles(smi)).apply(lambda m: GetScaffoldForMol(m)).apply(lambda scaff: MakeScaffoldGeneric(scaff)).apply(lambda scaff: GetMorganFingerprintAsBitVect(scaff, 3))
                    frame['murcko_fingerprint'] = frame['ROMol'].swifter.apply(lambda m: GetScaffoldForMol(m)).apply(lambda scaff: MakeScaffoldGeneric(scaff)).apply(lambda scaff: GetMorganFingerprintAsBitVect(scaff, 3))
                    
                    elapsed = time.time() - start

                st.info(f"__Murcko Fingerprint__ `{round(elapsed,2)}`s left...")

                ## Get Projection
                if st.session_state.projection:
                    with st.spinner('⌛ Processing __Murcko SMILES FP__ Projection...'): 
                        start = time.time()
                        frame = GetProjection(frame, 'murcko_fingerprint')
                        elapsed = time.time() - start
                    st.info(f"__Murcko Fingerprint Projection__ `{round(elapsed,2)}`s left...")
              
            
            # ImageMol FP section
            ## Get Feature
            if 'ImageMol' in st.session_state.feature_type:
                with st.spinner('⌛ Processing __ImageMol FP__...'):
                    frame.to_csv('./smiles.csv')
                    start = time.time()
                    # frame = GetImageMolFP(frame)
                    
                    # TODO: refactor
                    command = "docker run -it --rm --gpus 'device=6' --shm-size=8gb -v /home/owen/Code/MolMolParadise:/data imagemol --input /data/smiles.csv"
                    process = subprocess.Popen(command, shell=True)
                    process.wait()
                    imagemol_arr = np.load('./ImageMol/ImageMol_dict.npy',  allow_pickle=True)
                    imagemol_list = imagemol_arr.tolist()
                    _ = []
                    for idx, (key, value) in enumerate(imagemol_list.items()):
                        _.append({'SMILES': key, 'imagemol_features': np.array(value, dtype='f')})
                    imagemol_df = pd.DataFrame(_)
                    ##

                    elapsed = time.time() - start

                st.info(f"__ImageMol FP__ `{round(elapsed,2)}`s left...")
            
                ## Get Projection
                if st.session_state.projection:
                    with st.spinner('⌛ Processing __ImageMol FP__ Projection...'): 
                        start = time.time()
                        # frame = GetProjection(frame, 'imagemol_features')
                        
                        # TODO: refactor
                        features = np.array(imagemol_df['imagemol_features'].values.tolist()).squeeze() 

                        reducer = umap.UMAP(random_state=42, n_jobs=32)
                        embed = reducer.fit_transform(features)
                        embed_df = pd.DataFrame(embed, columns=[f'imagemol_features_x', f'imagemol_features_y'])

                        frame = pd.concat([frame, embed_df], axis=1)
                        ###################
                        
                        elapsed = time.time() - start
                    st.info(f"__ImageMol Projection__ `{round(elapsed,2)}`s left...")
                    
        with st.container():
            with st.spinner('⌛ Plot molecule images ...'): 
                frame['image'] = frame['SMILES'].apply(smi2svg)
                
            selection = dict(zip(st.session_state.class_name,st.session_state.select_idx))
            
            if 'SMILES Morgan FP' in st.session_state.feature_type:
                st.header('SMILES Morgan FP')
                # st.write(frame.head(5))
                DrawUMAP(frame, 'fingerprint_x', 'fingerprint_y', selection, 'SMILES Morgan FP')
            
            if 'Murcko Morgan FP' in st.session_state.feature_type:
                st.header('Murcko Morgan FP')
                # st.write(frame.head(5))
                DrawUMAP(frame, 'murcko_fingerprint_x', 'murcko_fingerprint_y', selection, 'Murcko Morgan FP')
                
            if 'ImageMol' in st.session_state.feature_type:
                st.header('ImageMol')
                DrawUMAP(frame, 'imagemol_features_x', 'imagemol_features_y', selection, 'ImageMol')
            
#             with st.spinner('⌛ Processing SMILES & Murcko FP...'):
#                 frame['fingerprint'] = frame['ROMol'].apply(GetFingerprint)
#                 frame['image'] = frame['ROMol'].apply(mol2svg)
#                 frame['murcko_generic'] = frame['smiles'].apply(GetGenericMurckoSmiles)
#                 frame['murcko_fingerprint'] = frame['murcko_generic'].apply(GetFingerprint)
#                 frame['murcko_image'] = frame['murcko_generic'].apply(mol2svg, args=(True,))

#                 reducer = umap.UMAP(
#                     random_state=42,
#                     # n_neighbors=10, 
#                     # min_dist=0.3,
#                 )

#                 murcko_reducer = umap.UMAP(
#                     random_state=42,
#                     # n_neighbors=10, 
#                     # min_dist=0.3,
#                 )

#                 ecfp = np.array(frame['fingerprint'].values.tolist()).squeeze() 
#                 embed = reducer.fit_transform(ecfp)
#                 embed_df = pd.DataFrame(embed, columns=['x', 'y'])

#                 murcko_ecfp = np.array(frame['murcko_fingerprint'].values.tolist()).squeeze()
#                 murcko_embed = murcko_reducer.fit_transform(murcko_ecfp)
#                 murcko_embed_df = pd.DataFrame(murcko_embed, columns=['murcko_x', 'murcko_y'])

#                 _frame = pd.concat([frame, embed_df, murcko_embed_df], axis=1)
#                 sample = _frame[['id', 'x', 'y', 'murcko_x', 'murcko_y', 'image', 'murcko_image', 'smiles']]
#                 sample['hit'] = sample['id'].apply(GetHits, args=(id_list,))

#                 sample.to_csv('./smiles.csv')

#             with st.spinner('⌛ Processing ImageMol...'):
#                 command = "docker run -it --gpus 'device=6' --shm-size=8gb -v /home/owen/Code/Test:/data imagemol --input /data/smiles.csv"
#                 process = subprocess.Popen(command, shell=True)
#                 process.wait()

#                 imagemol_arr = np.load('./ImageMol/ImageMol_dict.npy',  allow_pickle=True)
#                 imagemol_list = imagemol_arr.tolist()
#                 _ = []
#                 for idx, (key, value) in enumerate(imagemol_list.items()):
#                     # if idx > 2: break
#                     _.append({'smiles': key, 'features': np.array(value)})
#                 imagemol_df = pd.DataFrame(_)
#                 imagemol_reducer = umap.UMAP(random_state=42, n_jobs=8)

#                 imagemol_features = np.array(imagemol_df['features'].values.tolist()).squeeze() # TODO: re-write this one, ref: https://iwatobipen.wordpress.com/2019/02/08/convert-fingerprint-to-numpy-array-and-conver-numpy-array-to-fingerprint-rdkit-memorandum/
#                 imagemol_embed = reducer.fit_transform(imagemol_features)
#                 imagemol_embed_df = pd.DataFrame(imagemol_embed, columns=['imagemol_x', 'imagemol_y'])
#                 df = pd.concat([sample, imagemol_embed_df], axis=1)
#                 df.to_csv('test.csv')

#             hits = df.query('hit == True')
#             mols = df.query('hit == False')

#             st.header('Results')
#             # if st.button('Download CSV'):
#             #     href = download_file()
#             #     st.markdown(href, unsafe_allow_html=True)

#             from bokeh.layouts import row

#             mol_src_1 = ColumnDataSource(df[df['hit'] == False])
#             hit_src_1 = ColumnDataSource(df[df['hit'] == True])

#             mol_src_2 = ColumnDataSource(df[df['hit'] == False])
#             hit_src_2 = ColumnDataSource(df[df['hit'] == True])

#             mol_src_3 = ColumnDataSource(df[df['hit'] == False])
#             hit_src_3 = ColumnDataSource(df[df['hit'] == True])

#             TOOLTIPS = [
#                 ("ID", "@id"),
#                 ("Mol", "@image{safe}"),
#                 ("Murcko", "@murcko_image{safe}")
#             ]

#             # output_file(filename='../outputs/1_umap/adenosine_focused_lib_murcko_umap.html', title='ATPase focused lib UMAP')
#             p = figure(title='UMAP of Helicase Focuses Library in Murcko SMILES Form')
#             p.circle(
#                 x='murcko_x', y='murcko_y',
#                 source=mol_src_1,
#                 color='orange',
#                 legend_label='Total mols'
#             )
#             p.circle(
#                 x='murcko_x', y='murcko_y',
#                 source=hit_src_1,
#                 color='green',
#                 legend_label='Selected mols'
#             )
#             p.add_tools(HoverTool(tooltips=TOOLTIPS))
#             st.bokeh_chart(p, use_container_width=False)
#             # save(p)

#             # output_file(filename='../outputs/1_umap/adenosine_focused_lib_smiles_umap.html', title='Adenosine Focuses Library UMAP')
#             q = figure(title='UMAP of Helicase Focuses Library in SMILES Form')
#             q.circle(
#                 x='x', y='y',
#                 source=mol_src_2,
#                 color='orange',
#                 legend_label='Total mols'
#             )
#             q.circle(
#                 x='x', y='y',
#                 source=hit_src_2,
#                 color='green',
#                 legend_label='Selected mols'
#             )
#             q.add_tools(HoverTool(tooltips=TOOLTIPS))
#             st.bokeh_chart(q, use_container_width=False)
#             # save(q)

#             # output_file(filename='../outputs/1_umap/adenosine_focused_lib_imagemol_umap.html', title='Adenosine Focuses Library UMAP')
#             r = figure(title='UMAP of Adenosine Focuses Library in ImageMol')
#             r.circle(
#                 x='imagemol_x', y='imagemol_y',
#                 source=mol_src_3,
#                 color='orange',
#                 legend_label='Total mols'
#             )
#             r.circle(
#                 x='imagemol_x', y='imagemol_y',
#                 source=hit_src_3,
#                 color='green',
#                 legend_label='Selected mols'
#             )
#             r.add_tools(HoverTool(tooltips=TOOLTIPS))
#             # save(r)

#             st.bokeh_chart(r, use_container_width=False)
        
else:
    st.button('🚫 _Not ready for **shabu-shabu**_ !', disabled=True)


