"""
Analysis workflows
"""
from pathlib import Path
import pandas as pd
import numpy as np
from pathlib import Path
import os
from nilearn.plotting import plot_stat_map
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.reporting import make_glm_report, get_clusters_table
import matplotlib.pyplot as plt
from bids.layout import BIDSLayout
from nilearn import image as nimg
from nilearn.glm import threshold_stats_img
import seaborn as sns



def create_design_matrix(sub,run,run_dir):
    tr =2.0
    n_scans = 199
    frame_times = np.linspace(0, (n_scans - 1) * tr, n_scans)
    fmriprep_dir = path_to_fmriprep
    layout  = BIDSLayout(fmriprep_dir, validate=False,
                            config=['bids','derivatives'])
    confound_files = layout.get(subject=sub,datatype='func',desc='confounds',
                            extension="tsv",return_type='file')

    IM_event_dir = 'path_to_event_file/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.csv'
    IM_event_file = pd.read_csv(IM_event_dir, delimiter=',')

    X1 = make_first_level_design_matrix(
            frame_times,
            IM_event_file,
            drift_model=None,
            hrf_model='glover',
        )
 
    X1 = X1.drop(['constant'],axis=1).reindex(columns=['top-down','bottom-up','posterior'])

    arg = run-1
    confound_file = confound_files[arg]
    confound_df = pd.read_csv(confound_file, delimiter='\t')
    # Select confounds
    final_confounds = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'] + \
                      [f'a_comp_cor_{i:02d}' for i in range(5)] +['framewise_displacement']

    if confound_df.isnull().values.any():
        print("Warning: NaNs present in confound variables, replacing with zeros.")
    confounds_matrix = np.nan_to_num(confound_df[final_confounds].values)
    
    event_dir = 'path_to_event_timing/sub-'+str(sub)+'/sub-'+str(sub)+'_task-bayes_run-00'+str(run)+'_events.csv'
    event_file = pd.read_csv(event_dir, delimiter=',')


    X_timing = make_first_level_design_matrix(
        frame_times,
        event_file,
        hrf_model='glover',
        add_regs=confounds_matrix,
        add_reg_names=final_confounds,
        high_pass=0.0078,
    )

    X = pd.concat([X1,X_timing],axis=1)
    
    
    plot_design_matrix(X)

    plt.subplots_adjust(left=0.08, top=0.9, bottom=0.21, right=0.96, wspace=0.3)
    plt.savefig(os.path.join(run_dir,f'design_matrix_run{run}.jpg'))
    plt.show()
    
    dataplot = sns.heatmap(X.corr(), cmap="YlGnBu")
    plt.savefig(os.path.join(run_dir,f'regression_correlation_run{run}.jpg'))

    return X

def clean(sub,run):
    fmriprep_dir = '/bucket/DoyaU/Shuhei/cat_fox/fMRI/fmriprep_23'
    layout = BIDSLayout(fmriprep_dir,validate=False,
                                config=['bids','derivatives'])

    func_files = layout.get(subject=sub, datatype='func',desc='preproc',
                            space='MNI152NLin2009cAsym',extension='nii.gz',return_type='file')

    mask_files = layout.get(subject=sub,datatype='func',desc='brain',
                            space='MNI152NLin2009cAsym',extension='nii.gz',return_type='file')

    arg = run-1
    func_file = func_files[arg]
    mask_file = mask_files[arg]

    raw_func_img = nimg.load_img(func_file)

    func_img = raw_func_img

    # Set some constants
    high_pass = 0.0078
    t_r = 2

    #Clean! without confounds
    clean_img = nimg.clean_img(func_img,detrend=True,standardize=True,high_pass=high_pass,
                                t_r=t_r,mask_img=mask_file)

    clean_img = nimg.smooth_img(clean_img, 8)

    return clean_img, mask_file

def pad_vector(contrast_, n_columns):
    """Append zeros in contrast vectors."""
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

def create_contrast_vector_for_runs(contrast_dict, design_matrices):
    """Create a list of contrast vectors, one for each design matrix."""
    contrast_vectors = []
    
    for design_matrix in design_matrices:
        contrast_vector = np.zeros(design_matrix.shape[1])
        
        for regressor_name, contrast_value in contrast_dict.items():
            if regressor_name in design_matrix.columns:
                # Find the column index of the regressor and set the contrast value
                column_index = design_matrix.columns.get_loc(regressor_name)
                contrast_vector[column_index] = contrast_value
            else:
                raise ValueError(f"Regressor '{regressor_name}' not found in design matrix.")
        
        contrast_vectors.append(contrast_vector)
    
    return contrast_vectors

DATA_ITEMS = ['bold', 'mask', 'events', 'regressors', 'tr']

def first_level_wf(sub,in_files, output_dir, name='wf_1st_level'):

    os.makedirs(output_dir, exist_ok=True)


    fmri_img = []
    design_matrices = []
    for run in range(1,6):
        run_dir = Path(output_dir , 'run-' + str(run))
        if not Path(run_dir).exists():
            os.makedirs(run_dir)

        X = create_design_matrix(sub,run,run_dir)

        clean_img, mask = clean(sub, run)
        print('clean img: ',clean_img.shape)

        fmri_img.append(clean_img)
        design_matrices.append(pd.DataFrame(X))
    
    fmri_glm = FirstLevelModel(mask_img=mask, minimize_memory=False, signal_scaling=False,smoothing_fwhm=None)
    fmri_glm = fmri_glm.fit(fmri_img,design_matrices=design_matrices)

    n_columns = design_matrices[0].shape[1]

    contrasts = {
    'Top-down': {'top-down': 1},
    'Bottom-up': {'bottom-up': 1},
    'Posterior': {'posterior': 1},
    # Add more contrasts as needed
    }

    
    # for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    for contrast_id, contrast_dict in contrasts.items():
        contrast_vector = create_contrast_vector_for_runs(contrast_dict, design_matrices)
        z_map = fmri_glm.compute_contrast(contrast_vector,output_type='z_score')
        z_map.to_filename(Path(output_dir, f'{contrast_id}_z_map.nii.gz'))

        beta_map = fmri_glm.compute_contrast(contrast_vector,output_type='effect_size')
        beta_map.to_filename(Path(output_dir, f'{contrast_id}_beta_map.nii.gz'))

        plot_stat_map(
            z_map, threshold=3.0,
            title=f'{contrast_id}, fixed effects')
        plt.savefig(os.path.join(output_dir,f'{contrast_id}, fixed effect.jpg'))

        clean_map, threshold = threshold_stats_img(
            z_map, alpha=0.05,height_control='fdr',cluster_threshold=10
        )

        plot_stat_map(
            clean_map,
            threshold=threshold,
            black_bg=True,
            title=f"{contrast_id}(fdr=0.05), clusters > 10 voxels",
        )
        plt.savefig(os.path.join(output_dir, f"{contrast_id}(clusters > 10 voxels).jpg"))

        print(threshold)
        table = get_clusters_table(z_map, stat_threshold=threshold, cluster_threshold=10)
        table.set_index("Cluster ID", drop=True)
        print(table.head())

    report = make_glm_report(fmri_glm,
                             contrasts,
                             )
    report.save_as_html(os.path.join(output_dir,'report.html'))

    return


import sys
import logging
from pathlib import Path
from templateflow.api import get as tpl_get, templates as get_tpl_list
import nipype.interfaces.io as nio

__version__='1.0.0'
logging.addLevelName(25, 'IMPORTANT')
logging.addLevelName(15, 'VERBOSE')  # Add a new level between INFO and DEBUG
logger = logging.getLogger('cli')

metadata = {
    'Name': 'cat and fox',
    'BIDSVersion': '1.4.1',
    'PipelineDescription': {
        'Name': 'post-fMRIPrep-analysis'
    },
    'OriginalCodeURL': 'https://github.com/poldracklab/ds003-post-fMRIPrep-analysis'
}
   
#define subject group labels
group_label = ['ctl','scz']

def trim_subid(subj_array):
    for ss in range(len(subj_array)):
        #trim 'sub-' prefix
        subj_array[ss] = subj_array[ss][4:]
    return subj_array


def get_parser():
    """Define the command line interface"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description='Subject-level Model-based Analysis Workflow',
                            formatter_class=RawTextHelpFormatter)

    # Arguments as specified by BIDS-Apps
    # required, positional arguments
    # IMPORTANT: they must go directly with the parser object
    parser.add_argument(
        'derivatives_dir', action='store', type=Path,
        help='the root folder of a derivatives set generated with fMRIPrep '
             '(sub-XXXXX folders should be found at the top level in this folder).')
    parser.add_argument('output_dir', action='store', type=Path,
                        help='the output path for the outcomes of preprocessing and visual '
                             'reports')
    parser.add_argument('analysis_level', choices=['run', 'subject','sub_higher', 'group', 'group-diff'], nargs='+',
                        help='processing stage to be run, "run" means run analysis of each subject, "subject" means individual  (combine runs) '
                             ', "group" is second level analysis.')

    parser.add_argument('--version', action='version', version=__version__)

    # Options that affect how pyBIDS is configured
    g_bids = parser.add_argument_group('Options for filtering BIDS queries')
    g_bids.add_argument('--participant-label', action='store', type=str,
                        nargs='*', help='process only particular subjects')
    g_bids.add_argument('--task', action='store', type=str, nargs='*',
                        help='select a specific task to be processed')
    g_bids.add_argument('--run', action='store', type=int, nargs='*',
                        help='select a specific run identifier to be processed')
    g_bids.add_argument('--space', action='store', choices=get_tpl_list() + ['T1w', 'template'],
                        help='select a specific space to be processed')
    g_bids.add_argument('--bids-dir', action='store', type=Path,
                        help='point to the BIDS root of the dataset from which the derivatives '
                             'were calculated (in case the derivatives folder is not the default '
                             '(i.e. ``BIDS_root/derivatives``).')

    g_perfm = parser.add_argument_group('Options to handle performance')
    g_perfm.add_argument("-v", "--verbose", dest="verbose_count", action="count", default=0,
                         help="increases log verbosity for each occurence, debug level is -vvv")
    g_perfm.add_argument('--ncpus', '--nprocs', action='store', type=int,
                         help='maximum number of threads across all processes')
    g_perfm.add_argument('--nthreads', '--omp-nthreads', action='store', type=int,
                         help='maximum number of threads per-process')

    g_other = parser.add_argument_group('Other options')

    return parser


def main():
    """Entry point"""
    from os import cpu_count
    from multiprocessing import set_start_method
    from bids.layout import BIDSLayout
    from nipype import logging as nlogging
    set_start_method('forkserver')
    

    opts = get_parser().parse_args()

    # Retrieve logging level
    log_level = int(max(25 - 5 * opts.verbose_count, logging.DEBUG))
    # Set logging
    logger.setLevel(log_level)
    nlogging.getLogger('nipype.workflow').setLevel(log_level)
    nlogging.getLogger('nipype.interface').setLevel(log_level)
    nlogging.getLogger('nipype.utils').setLevel(log_level)

    # Resource management options
    plugin_settings = {
        'plugin': 'MultiProc',
        'plugin_args': {
            'n_procs': opts.ncpus,
            'mem_gb' : 8,
            'raise_insufficient': True,
            'maxtasksperchild': 1,
        }
    }

    # Permit overriding plugin config with specific CLI options
    if not opts.ncpus or opts.ncpus < 1:
        plugin_settings['plugin_args']['n_procs'] = cpu_count()

    nthreads = opts.nthreads
    if not nthreads or nthreads < 1:
        nthreads = cpu_count()

    derivatives_dir = opts.derivatives_dir.resolve()
    bids_dir = opts.bids_dir or derivatives_dir.parent

    # get absolute path to BIDS directory
    bids_dir = opts.bids_dir.resolve()
    print(bids_dir)
    print(str(derivatives_dir))
    layout = BIDSLayout(str(bids_dir), validate=False, derivatives=derivatives_dir)
    query = {'scope': 'derivatives', 'desc': 'preproc',
             'suffix': 'bold', 'extension': ['.nii', '.nii.gz']}
    print(layout)

    if opts.participant_label:
        query['subject'] = '|'.join(opts.participant_label)
    if opts.run:
        query['run'] = '|'.join(opts.run)
    if opts.task:
        query['task'] = '|'.join(opts.task)
    if opts.space:
        query['space'] = opts.space
        if opts.space == 'template':
            query['space'] = get_tpl_list()

    # Preprocessed files that are input to the workflow
    prepped_bold = layout.get(**query)
    if not prepped_bold:
        print('No preprocessed files found under the given derivatives '
              'folder "%s".' % derivatives_dir, file=sys.stderr)

    base_entities = set(['subject', 'session', 'task', 'run', 'acquisition', 'reconstruction'])
    inputs = {}

    for part in prepped_bold:
        entities = part.entities
        sub = entities['subject']
        run = entities['run']
        if not inputs.get(sub):
            inputs[sub] = {}
        inputs[sub][run] = {}
        base = base_entities.intersection(entities)
        subquery = {k: v for k, v in entities.items() if k in base}
        inputs[sub][run]['bold'] = part.path
        inputs[sub][run]['mask'] = layout.get(
            scope='derivatives',
            suffix='mask',
            return_type='file',
            extension=['.nii', '.nii.gz'],
            space=query['space'],
            **subquery)[0]
        inputs[sub][run]['events'] = layout.get(
            suffix='events', return_type='file', **subquery)[0]
        inputs[sub][run]['regressors'] = layout.get(
            scope='derivatives',
            suffix='timeseries',
            return_type='file',
            extension=['.tsv'],
            **subquery)[0]
        inputs[sub][run]['anat'] = layout.get(
            scope='derivatives',
            suffix='T1w',
            return_type='file',
            extension=['.nii', '.nii.gz'],
            space=query['space'],
            subject=subquery['subject'])[0]
        inputs[sub][run]['tr'] = entities['RepetitionTime']

    sub_list = sorted(inputs.keys())
    subject = sub_list[0]

    output_dir = opts.output_dir.resolve() 

    if 'run' in opts.analysis_level:
        
        logger.info('Writting 1st level outputs to "%s".', output_dir)

        output_dir_sub = Path(output_dir, subject)
        print(output_dir_sub)
        
        first_level_wf(subject,inputs,output_dir_sub)


if __name__=='__main__':
    sys.exit(main())