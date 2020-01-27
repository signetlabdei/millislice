import sem
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tikzplotlib # Save figures as PGFplots

#from statistics import mean
# from pympler import tracker

# Functions

def plot_forall_static(static, param_ca, param_no_ca, versus, fewer_images=False):
    # tr = tracker.SummaryTracker()
    # Get list of values of params that are suppoSsed to be static
    campaign = sem.CampaignManager.load('./slicing-res')
    loaded_params = campaign.db.get_all_values_of_all_params()
    # For now, just one..
    static_values = loaded_params[static]
    counter = 0
    for val in static_values:
        counter = counter + 1
        # Restrict params
        param_ca[static] = val
        param_no_ca[static] = val
        # Ugly, but lazy
        metric_bucket_dummy = {'versus':val}
        static_formatted = sanitize_versus(vs=static, metric_bucket=metric_bucket_dummy)
        val_formatted = metric_bucket_dummy['versus']
        # Save the plot
        out_dir = f"./slicing-plots/versus_{versus}/{val_formatted}/"
        os.makedirs(out_dir, exist_ok=True)
        # Use lower level functions to plot
        fig = plot_all_metrics(param_no_ca=param_no_ca, param_ca=param_ca, versus=versus,
                            fewer_images=fewer_images, top_path=out_dir)

        if fewer_images:
            fig.suptitle(f"System performance for {static_formatted} = {val_formatted}", fontsize=16)
            plt.savefig(f"{out_dir}System_performance.png" )
            tikzplotlib.save(f"{out_dir}System_performance.tex")
            plt.close('fig')

        #tr.print_diff()
        print(f"{counter/len(static_values)*100:.0f} % done!")

def plot_all_metrics(param_ca, param_no_ca, versus=None, fewer_images=False, top_path=None):

    if fewer_images:
        fig, ax = plt.subplots(constrained_layout=True, nrows=2, ncols=3)        
    
    campaign = sem.CampaignManager.load('./slicing-res')
    trace_str_rx = 'test_RxPacketTrace.txt' # Always same name
    trace_err = 'stderr'
    band = pd.DataFrame()
    #d = {'embb_thr' : [0], 'urllc_loss' : [0]}
    thr_embb = pd.DataFrame(columns=['thr', 'ccMan', 'mode', 'runSet', versus])
    delay_urllc = pd.DataFrame(columns=['delay', 'ccMan', 'mode', 'runSet', versus])

    for prot in ['URLLC', 'eMBB']:
        thr = []
        delay = []
        loss = []

        # Select proper trace file
        if prot == 'URLLC':
            trace_str_dl = 'test_urllc-dl-app-trace.txt'
            trace_str_ul = 'test_urllc-ul-sink-app-trace.txt'
            sub_col = 0
        elif prot == 'eMBB':
            trace_str_dl = 'test_eMBB-dl-app-trace.txt'
            trace_str_ul = 'test_eMBB-ul-app-trace.txt'
            sub_col = 1

        for param in param_ca, param_no_ca:
            print(f"\t {prot} stats, params {param}")
            err_amount = 0
            # Load the desired datasets
            # Load results, specify params if given on input
            # Get the required files IDs
            if param is not None:
                res_data = campaign.db.get_results(param)
            else:
                res_data = campaign.db.get_results()
            for res_istance in res_data:
                res_id = res_istance['meta']['id']
                # Load all the desired traces
                dl_path = campaign.db.get_result_files(res_id)[trace_str_dl]
                ul_path = campaign.db.get_result_files(res_id)[trace_str_ul]
                rx_path = campaign.db.get_result_files(res_id)[trace_str_rx]
                err_path = campaign.db.get_result_files(res_id)[trace_err]

                # Save both results and relative params
                dl_df = pd.read_csv(filepath_or_buffer=dl_path, header=0, delimiter='\t', low_memory=False)
                ul_df = pd.read_csv(filepath_or_buffer=ul_path, header=0, delimiter='\t', low_memory=False)
                rx_df = pd.read_csv(filepath_or_buffer=rx_path, header=0, delimiter='\t', low_memory=False)
                # Did ns3 crash/output some error?
                with open(err_path, 'r') as file:
                    errors = file.read()
                if len(errors) != 0:
                    # Skip usch trace
                    err_amount = err_amount + 1
                    continue # Check other traces

                # Improve data structure, keep just relevant data
                ul_df = sanitize_dataframe(ul_df, res_istance['params']['maxStart']*1e9) # sec to ns ns in the traces
                dl_df = sanitize_dataframe(dl_df, res_istance['params']['maxStart']*1e9) # sec to ns ns in the traces
                rx_df = sanitize_dataframe(rx_df, res_istance['params']['maxStart']*1e9) # sec to ns ns in the traces

                params = res_istance['params']
                # Compute metrics here
                run_loss = pkt_loss_app(dl_df, ul_df)
                loss.append({'mean': run_loss, 'params': params})

                run_thr = throughput_app(dl_df, bearer_type=prot, params=params)
                thr.append({'mean':run_thr, 'params': params})
                if prot == 'eMBB':
                    thr_embb.loc[len(thr_embb)] = {'thr':run_thr*params['numEmbbUes'], 'ccMan':params['ccMan'],
                     'mode':params['mode'], 'runSet':params['runSet'], versus:params[versus]}

                run_delay = delay_app(dl_df)
                delay.append({'mean':run_delay, 'params': params})
                if prot == 'URLLC':
                    delay_urllc.loc[len(delay_urllc)] = {'delay':run_delay, 'ccMan':params['ccMan'],
                     'mode':params['mode'], 'runSet':params['runSet'], versus:params[versus]}

                band = band.append(band_allocation(rx_df, versus, res_istance['params']))

            
            # If no valid trace loaded, raise an error
            if err_amount >> 0:
                print(f"\t---{err_amount} faulty traces found, out of {len(res_data)}---")
            if err_amount >= len(res_data):
                print('Unusable traces, perform other simulations!')


                
        # Plot the various metrics 
        if fewer_images:
            info = {'prot':prot, 'metric':'Delay', 'unit':'[ms]'}       
            plot_lines_versus(metric_bucket=delay, info=info, s_path=top_path, versus=versus, fig=fig, ax=ax[sub_col, 2])
            info = {'prot':prot, 'metric':'Throughput', 'unit':'[Mbit/s]'}
            plot_lines_versus(metric_bucket=thr, info=info, s_path=top_path, versus=versus, fig=fig, ax=ax[sub_col, 1])
            info = {'prot':prot, 'metric':'Packet loss', 'unit':''}
            plot_lines_versus(loss, s_path=top_path, info=info, versus=versus, fig=fig, ax=ax[sub_col, 0])
        else:
            info = {'prot':prot, 'metric':'Delay', 'unit':'[ms]', 'path':top_path}       
            plot_lines_versus(metric_bucket=delay, info=info, s_path=top_path, versus=versus)
            info = {'prot':prot, 'metric':'Throughput', 'unit':'[Mbit/s]', 'path':top_path}
            plot_lines_versus(metric_bucket=thr, info=info, s_path=top_path, versus=versus)
            info = {'prot':prot, 'metric':'Packet loss', 'unit':'', 'path':top_path}
            plot_lines_versus(loss, s_path=top_path, info=info, versus=versus)


            
    # Band allocation plot here
    m_title = 'Band allocation \n (percentage of total system bw)'
    plot_metric_box(band, s_path=top_path, metric='Band allocation', title=m_title, versus=versus)
    # Thr vs delay
    
    plot_scatter(delay=delay_urllc, thr=thr_embb, versus=versus, s_path=top_path)
    
    if fewer_images:
        return fig
    else:
        return None

def save_fig(fig, info):
    plt.title(f"{info['prot']} average {info['metric']} ", fontsize=12)
    plt.savefig(f"{info['path']}{info['metric']}_{info['prot']}.png" )
    tikzplotlib.save(f"{info['path']}{info['metric']}_{info['prot']}.tex")
    plt.close('fig')

def group_cc_strat(metric_frame):
    metric_frame['mode'] =  metric_frame['mode'].replace(1, 'no CA, ')
    metric_frame['mode'] =  metric_frame['mode'].replace(2, 'CA, ')
    metric_frame['ccMan'] =  metric_frame['ccMan'].replace(0, 'SplitDrb')
    metric_frame['ccMan'] =  metric_frame['ccMan'].replace(1, 'SlicingDrb')
    metric_frame['ccMan'] =  metric_frame['ccMan'].replace(2, 'placeholder')

    metric_frame['CC strategy'] = metric_frame['mode'] + metric_frame['ccMan']
    metric_frame['CC strategy'] =  metric_frame['CC strategy'].replace('no CA, SplitDrb', 'no CA')
    metric_frame['CC strategy'] =  metric_frame['CC strategy'].replace('CA, placeholder', 'CA')

    return metric_frame

def plot_scatter(delay, thr, versus, s_path):

    delay = group_cc_strat(delay)
    delay.drop(['ccMan', 'mode'], axis=1, inplace=True)
    delay.sort_values(by=['CC strategy', versus, 'runSet'], inplace=True)
    thr = group_cc_strat(thr)
    thr.drop(['ccMan', 'mode'], axis=1, inplace=True)
    thr.sort_values(by=['CC strategy', versus, 'runSet'], inplace=True)

    delay['runSet'] = thr['thr']
    delay.rename(columns = {'runSet':'thr', versus:'versus'}, inplace = True)
    out_str = sanitize_versus(versus, delay)
    test = delay.groupby(['CC strategy', 'versus'],as_index=False).mean()    

    fig, ax = plt.subplots(constrained_layout=True)
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    ax = sns.scatterplot(data=test, x='thr', y='delay', hue='CC strategy', size='versus', sizes=(100, 200))

    # Set graphical properties, title and filename
    ax.set_xlabel(f"{out_str}", fontsize=12)
    ax.set_ylabel(f"URLLC delay [ms]", fontsize=12)
    plot_title = f"Throughput eMBB vs delay URLLC"
    handles, labels = ax.get_legend_handles_labels()

    max = len(set(test['CC strategy'])) + 1
    ax.legend(handles=handles[:max], labels=labels[:max], loc='best')

    fig.set_size_inches(7, 3)    
    plt.title(plot_title, fontsize=12)
    plt.savefig(f"{s_path}{plot_title}.png" )
    #tikzplotlib.save(f"{s_path}{plot_title}.tex")
    plt.close(fig)


def plot_lines_versus(metric_bucket, info, s_path, versus, fig=None, ax=None):

    dummy_ax = ax
    if ax is None:
        fig, ax = plt.subplots(constrained_layout=True)
        sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    metric_data = []
    mode_data = []
    ccman_data = []
    versus_data = []

    for res in metric_bucket:
        metric_data.append(res['mean'])
        mode_data.append(res['params']['mode'])
        ccman_data.append(res['params']['ccMan'])
        versus_data.append(res['params'][versus])

    frame = {
        'metric': metric_data,
        'mode': mode_data,
        'versus': versus_data,
        'ccMan': ccman_data
    }

    metric_frame = pd.DataFrame(data=frame)

    metric_frame = group_cc_strat(metric_frame)
    filename = f"{info['prot']}_{info['metric']}_vs{versus}.png"
    temp = sanitize_versus(metric_bucket=metric_frame, vs=versus)
    if temp is not None:
        versus = temp

    if len(set(metric_frame['CC strategy'])) == 4:
         h_ord = ['no CA', 'CA', 'CA, SplitDrb', 'CA, SlicingDrb']
    else:
        h_ord = ['no CA', 'CA, SplitDrb', 'CA, SlicingDrb']
    
    g = sns.lineplot(data=metric_frame, x='versus', y='metric', err_style='bars', 
                        hue='CC strategy', hue_order=h_ord, ax=ax)

    # Set graphical properties, title and filename
    ax.set_ylabel(f"{info['metric']} {info['unit']} \n", fontsize=12)
    ax.set_xlabel(f"{versus}", fontsize=12)
    plot_title = f"{info['metric']} {info['prot']} vs. {versus}"

    if info['metric'] == 'Throughput':
        _, top = ax.get_ylim()
        g.set(ylim=(0, top*1.1)) 

    if dummy_ax is None:
        fig.set_size_inches(count_amount_uniques(versus_data)*2, 3)    
        save_fig(fig, info)
        plt.close(fig)
        # Save, create dir if doesn't exist 
    else:
        fig.set_size_inches(count_amount_uniques(versus_data)*2*3, 5)
        ax.grid(color='#b3b3b3')
        ax.set_facecolor('#f5f5fa')
        ax.title.set_text(f"{plot_title} \n")
        for spine in ax.spines.values():
            spine.set_edgecolor('#b3b3b3')

    del dummy_ax
    del g
    #Ylim
    #g.set(ylim=(0, None)) 
'''
def plot_line(metric_frame, metric, title, s_path, fname, overlays=None):

    fig, ax = plt.subplots(constrained_layout=True)
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})
    color = '#7a4e4f'

    sns.lineplot(data=metric_frame, x='x', y='y', color=color)

    x_handles = []
    y_handles = []

    # Plot overlays if available
    if overlays is not None:
        for item in overlays['x']:
            ax.axvline(x=item, color=[0,0,0], linewidth=0.7, label='appEnd')
        for item in overlays['y']:
            ax.axhline(y=item, color=color, linestyle='dashed', linewidth=1, label='Average value of whole system')

    #f, axes = plt.subplots(1, 2)
    plt.ylabel(f"{metric} \n", fontsize=11)
    plt.title(title + '\n') 

    ax.legend(loc='best')

    out_dir = s_path
    os.makedirs(out_dir, exist_ok=True)
    fig.set_size_inches(8, 6)

    plt.savefig(out_dir + fname)
    plt.close('fig')
'''
def plot_distr_bins(metric_frame, metric, title, s_path):
    # Make sure figure is clean
    fig, ax = plt.subplots(constrained_layout=True)
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    sns.distplot(metric_frame, kde=False, norm_hist=True)

    plt.xlabel(f"{metric} \n", fontsize=11)
    plt.title(title + '\n') 

    fig.set_size_inches(8, 3)

    # Save, create dir if doesn't exist       
    out_dir = f"{s_path}detailed/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}{metric}")

    plt.close('fig')

def plot_metric_box(metric_frame, metric, title, s_path, versus):
    # Make sure figure is clean
    fig, ax = plt.subplots(constrained_layout=True)

    # Clean up and group by ccMan strategy
    metric_frame = group_cc_strat(metric_frame)
    x_label = sanitize_versus(vs=versus, metric_bucket=metric_frame)

    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

    # Plot sum as background
    metric_frame['band_alloc_cc1'] = metric_frame['band_alloc_cc1'] + metric_frame['band_alloc_cc0']

    if len(set(metric_frame['CC strategy'])) == 4:
         h_ord = ['no CA', 'CA', 'CA, SplitDrb', 'CA, SlicingDrb']
    else:
        h_ord = ['no CA', 'CA, SplitDrb', 'CA, SlicingDrb']


    sns.barplot(x='versus', y='band_alloc_cc1', hue='CC strategy', hue_order=h_ord, 
                data=metric_frame, palette=sns.color_palette('pastel'))
    # Plot cc0 on foreground
    ax_bckg = sns.barplot(x='versus', y='band_alloc_cc0', hue='CC strategy', 
                            hue_order=h_ord, data=metric_frame, palette=sns.color_palette('muted'))
    handles, labels = ax_bckg.get_legend_handles_labels()
    for dummy in range(0, 3):
        labels[dummy] = 'CC1 - ' +  labels[dummy] 
        labels[3 + dummy] = 'CC0 - ' +  labels[3 + dummy]
    ax_bckg.legend(handles=handles[1:], labels=labels[1:], ncol=2, loc='best')

    # ax_bckg.legend(handles=handles[3:], labels=labels[3:])
    
    # Title, labels ecc.
    fig.set_size_inches(6, 7)
    filename = f"{metric}.png"
    plt.ylabel(f"{metric} \n", fontsize=12)
    plt.xlabel(f"{x_label}", fontsize=12)
    #ax.set_xlabel('')
    plt.title(f"{title} \n")

    # Save, create dir if doesn't exist       
    out_dir = s_path
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}{filename}")
    tikzplotlib.save(f"{out_dir}{metric}.tex")

    plt.close(fig)

def plot_metrics_generic(metric_bucket, metric, prot, s_path, unit, vs=None):
    """ 
    Plots metric mean, CI and all run samples
    Args:
        versus (str): param to use on the x axis
    """
    # Make sure figure is clean
    plt.clf()

    # Build dataframe
    metric_data = []
    mode_data = []
    versus_data = []

    for res in metric_bucket:
        metric_data.append(res['mean'])
        mode_data.append(res['params']['mode'])
        versus_data.append(res['params'][vs])

    frame = {
        'metric': metric_data,
        'mode': mode_data,
        'versus': versus_data
    }

    metric_frame = pd.DataFrame(data=frame)
    metric_frame['mode'].replace(1, 'no CA',inplace=True)
    metric_frame['mode'].replace(2, 'CA', inplace=True)
    
    # Colors
    # fig, (box_ax, viol_ax) = plt.subplots(2, 1, constrained_layout=True, sharex=True, sharey=True)
    fig, ax = plt.subplots(constrained_layout=True)
    # Avoid having strange ticks formatting
    plt.ticklabel_format(useOffset=False, style='plain')

    dark_palette = ['#465782', '#7a4e4f']
    light_palette = ['#90a5e0', '#c27a7c']
    sns.set_style('whitegrid', {'axes.facecolor': '#EAEAF2'})

      # Save, with the proper size
    if check_constant(versus_data):
        filename = f"{prot}_{metric}_CA_vs_nonCA.png"
        plot_title = f"{prot} {metric}"
    else:
        filename = f"{prot}_{metric}_vs{vs}_CA_vs_nonCA.png"
        # viol_ax.set_xlabel(f"{vs}", fontsize=11)
        # box_ax.set_xlabel(f"{vs}", fontsize=11)


    # Violin plotax
    # sns.violinplot(data=metric_frame, y='metric', x='versus', hue='mode', palette=light_palette, split=True, inner='stick', ax=viol_ax)

    vs = sanitize_versus(metric_bucket=metric_frame, vs=vs)
    # Boxplot
    sns.boxplot(data=metric_frame, y='metric', x='versus', hue='mode', palette=light_palette, ax=ax)
    strip_handle = sns.stripplot(x="versus", y="metric", hue="mode", data=metric_frame, dodge=True, palette=dark_palette, ax=ax)

    # Remove legend duplicate
    handles, labels = strip_handle.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2])

    # Save, with the proper size
    if check_constant(versus_data):
         # Overlay the mean values
        # overlay_means(metric_bucket, palette=dark_palette, vs=vs, vs_data=versus_data)

        # Set graphical properties
        fig.set_size_inches(4, 4)
        # Set title and filename
        plot_title = f"{prot} {metric}"
    else:
        # Overlay the mean values, if traces are not empty
        # overlay_means(metric_bucket, palette=dark_palette, vs=vs, vs_data=versus_data)

        # Set graphical properties
        fig.set_size_inches(count_amount_uniques(versus_data)*2.5, 7.5)
         # Set title and filename
        plot_title = f"{prot} {metric} vs. {vs}"
        ax.set_xlabel(f"{vs}", fontsize=12)
        # viol_ax.set_xlabel(f"{vs}", fontsize=11)
        # box_ax.set_xlabel(f"{vs}", fontsize=11)
        

    ax.set_ylabel(f"{metric} {unit} \n", fontsize=12)
    fig.suptitle(plot_title + '\n', fontsize=12)

    # Save, create dir if doesn't exist       
    out_dir = f"./slicing-plots/{s_path}/"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(out_dir + filename)

    plt.close('fig')
'''
def overlay_means(metric_bucket, palette, vs, vs_data):

   # Overlay mean value
    # Obtain info regarding bounds
    left_orig, right_orig = plt.xlim()
    # Compute means
    means_bucket = compute_means(group_by_params(metric_bucket))
    # Get possible, different vs values
    vs_uniques = list(set(vs_data))
    vs_uniques.sort()

    for index in range(len(vs_uniques)):
        just_right_means = find_elements(means_bucket, vs, vs_uniques[index])
        # If no trace loaded for such param combination, do nothing
        if(len(just_right_means) >= 2):
            no_ca_mean = find_elements(just_right_means, 'mode', 1)[0]['mean']
            ca_mean = find_elements(just_right_means, 'mode', 2)[0]['mean']
            # Get plot limits and plot
            width = (right_orig - left_orig)/len(vs_uniques)
            left = left_orig + index*width
            right = left + width
            plt.plot([left, right], [no_ca_mean, no_ca_mean], color=palette[0], linestyle='--', linewidth=1)
            plt.plot([left, right], [ca_mean, ca_mean], color=palette[1], linestyle='--', linewidth=1)
'''

def sanitize_versus(vs, metric_bucket):
    if(vs == 'embbUdpIPI'):
        metric_bucket['versus'] = round(1024*8/(metric_bucket['versus']), 1) # packet_size*bits in a bye/rate
        return 'eMBB sources rate [Mbit/s]'  
    if(vs == 'urllcUdpIPI'):
        metric_bucket['versus'] = round(1024*8/(metric_bucket['versus']), 1) # packet_size*bits in a bye/rate
        return 'URLLC sources rate [Mbit/s]'
    if(vs == 'ccRatio'):
        return 'Ratio of bw allocated to CC0'
    if(vs == 'numEmbbUes'):
        return 'amount of eMBB users'
    if(vs == 'numUrllcUes'):
        return 'amount of URLLC users'

def find_elements(bucket, param, value):
    out = []
    for element in bucket:
        temp = element['params'][param]
        if temp == value:
            out.append(element)
    return out

def group_by_params(metric_bucket):
    # Remove param specifying different runs
    for sim in metric_bucket:
        sim['params'].pop('RngRun', None)
        sim['params'].pop('runSet', None)
    out_bucket = []
    # Group sims having same param
    while(len(metric_bucket) > 0):
        temp_param = metric_bucket[0]['params']
        mean_bucket = []
        waste_bucket = []
        # Find sims with same params
        for sim in metric_bucket:
            if sim['params'] == temp_param:
                mean_bucket.append(sim['mean'])
                waste_bucket.append(sim)
        # Create entry for same params, different runs
        out_bucket.append({
            'mean': mean_bucket,
            'params': temp_param
        })
        # Remove elements that have been joined
        for sim in waste_bucket:
            metric_bucket.remove(sim)

    return out_bucket

def load_results(trace_name, param=None):
    """ 
    Loads the results from specifics file traces.
    If params are specified, it loads just the results of the
    simulations corresponding to such mean.

    Args:
        trace_name (str): filename of the trace(s) to load

    Returns:
        A list of results matching the query, one entry for every simulation run. 
        The structure of the returning value is a list, comprised of dictionaries
        having keys 'results' and 'params'. The first holds an array of results     
        for every traced parameter, the latter the combination of params that 
        originated such outputs.
    """



    # Get list containing data of the trace for the various param combination
    # and combination of params that generated it
    res_bucket = []
    
    return res_bucket

def sanitize_dataframe(dataframe, treshold):
    # Remove trailing whitespaces from cols names
    dataframe = dataframe.rename(columns=lambda x: x.strip())
    # We want to keep trace just of packets transmitted after all apps started
    if 'tx_time' in dataframe.columns:
        dataframe = dataframe[dataframe['tx_time'] > treshold]   
    else:
        dataframe = dataframe[dataframe['time'] > treshold/1e9] # Need secs here

    # Remove unused cols
    if 'TBler' in dataframe.columns:
        dataframe.drop(columns=['TBler', 'corrupt', 'mcs', 'rv' , '1stSym', 'DL/UL' , 'time', 'tbSize'], inplace=True)

    return dataframe

def sinr_overall(trace_data):

    print('--Computing SINR statistics--')

    # Start by plotiing SINR distribution, averaged over the runs
    sinr_bucket = []

    for item in trace_data:
        sinr_bucket.extend(item['results']['SINR(dB)'])

    snr_frame = {
        'sinr': sinr_bucket
    }

    return pd.DataFrame(data=snr_frame)


def band_allocation(trace_data, versus, params):

    # Find out total amount symbols available
    avail_sym = trace_data['frame'].iloc[-1] - trace_data['frame'].iloc[0]
    avail_sym = avail_sym*10*22 # Frames*subframes in a frame*symbols in a subframe
    # Get info regarding first CC
    item_cc0 = trace_data[trace_data['ccId'] == 0]
    used_sym_cc0 = item_cc0['symbol#'].sum()

    # If we are using CA, get also info regarding secondary CC
    if params['mode'] == 2:
        item_cc1 = trace_data[trace_data['ccId'] == 1]
        used_sym_cc1 = item_cc1['symbol#'].sum()
        # Normalize usage for stacked 
        cc0 = (used_sym_cc0/avail_sym)*params['ccRatio']
        cc1 = (used_sym_cc1/avail_sym)*(1-params['ccRatio'])
    else:
        cc1 = 0
        cc0 = used_sym_cc0/avail_sym 

    frame = {
        'band_alloc_cc0': [cc0],
        'band_alloc_cc1': [cc1],
        'ccMan': [params['ccMan']],
        'mode': [params['mode']],
        'versus': [params[versus]]
    }

    out = pd.DataFrame.from_dict(data=frame)
    return out

def throughput_app_det(trace_data, bearer_type, vs, s_path):

    out = []
    versus_data = []
    # Compute and plot overall throughput in the worst case
    
    for sim in trace_data:
        versus_data.append(sim['params'][vs])

    # Obtain unique values
    versus_data = list(set(versus_data))

    for vs_value in versus_data:
        # Keep just traces using such value of the vs param
        temp_trace_data = []
        for item in trace_data:
            if (item['params'][vs] == vs_value):
                temp_trace_data.append(item)

        # Compute overall mean throughputs, pick the one with the worst average across the runs
        thr_bucket = throughput_app(temp_trace_data, bearer_type)
        mean_frame = []
        for item in thr_bucket:
            mean_frame.append(item['mean'])

        temp_frame = pd.DataFrame({
            'value': mean_frame
        })

        choosen_one = copy.deepcopy(trace_data[temp_frame['value'].idxmin()]['results'])
        # Compute throughput versus time
        choosen_one['rx_time'] = (choosen_one['rx_time']/10e8).round(2)
        packets_rx = choosen_one.groupby(['rx_time']).count()

        # Fix col names
        packets_rx['x'] = packets_rx.index.values
        packets_rx.rename(columns={'tx_time':'y'}, inplace=True)
        # Packets every 0.1s --> Mbit/s
        packets_rx['y'] = packets_rx['y']*1024*8/1e5 # 10e6 for bit to Mbit, then 0.1 to 1 s
        packets_rx.drop(columns=['pkt_size', 'seq_num', 'node_id'], inplace=True)

        temp_title = 'Throughput of whole system vs time [s]'
        temp_path =  s_path + 'detailed/'
        filename = f"{bearer_type}_{vs}_{vs_value}"

        # Overlay mean throughput and appEnd 
        overlays = {
            'x': [temp_trace_data[0]['params']['appEnd']],
            'y': [temp_frame['value'].min()]
        }

        plot_line(metric_frame=packets_rx, metric='Throughput [Mbit/s]', title=temp_title, 
                    s_path=temp_path, fname=filename, overlays=overlays)

    return out

def throughput_app(trace_data, bearer_type, params):
    """ 
    Computes the average throughput @ APP layer
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """

    full_regime_data = trace_data[trace_data['rx_time'] < params['appEnd']*1e9]
    g = (len(full_regime_data.index)*1024*8)/((params['appEnd'] -
                                        params['maxStart'])*1e6)  # computing overall throughput
    del full_regime_data
    # computing per user throughput
    if bearer_type == 'URLLC':
        single_g = g/(params['numUrllcUes'])
    else:
        single_g = g/(params['numEmbbUes'])

    return single_g

def delay_app(trace_data):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    time_rx = trace_data['rx_time']
    # get time of tx
    time_tx = trace_data['tx_time']
    # packet delay
    #'var': pck_delay.std(),  # Output both latency and jitter
    pck_delay = (time_rx - time_tx)/1e6
    return pck_delay.mean()

def pkt_loss_app(trace_dl, trace_ul):
    """ 
    Computes the average delay @ APP layer.
    If parameters combination are provided, then only the simulations for
    such mean are taken into consideration for the computation.

    Args:
        bearer_type (str): either urrlc or embb
    """
    loss = 0
    sent = len(trace_ul.index)
    # Overall lost packets
    dropped = sent - len(trace_dl.index)
    
    return dropped/len(trace_ul.index)

# Small, support functions
def check_constant(bucket):
    return bucket[1:] == bucket[:-1]

def count_amount_uniques(bucket):
    return len(set(bucket))

def compute_means(metric_bucket):
    # Save original data
    out_bucket = copy.deepcopy(metric_bucket)

    for index in range(len(metric_bucket)):
        out_bucket[index]['mean'] = mean(metric_bucket[index]['mean'])
    return out_bucket

# Actual metrics computation
# Try plot

print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs eMBB rates')
ca_params = {'f0': 28e9, 'f1':10e9,'mode': 2, 'ccRatio': 0.5,'numEmbbUes':10, 'numUrllcUes':10 }
no_ca_params = {'f0': 28e9, 'mode': 1, 'ccRatio': 0.5, 'ccMan':0, 'numEmbbUes':10, 'numUrllcUes':10}

print('Computing stats')
plot_forall_static(param_ca=ca_params, param_no_ca=no_ca_params, versus='embbUdpIPI', fewer_images=False, static='urllcUdpIPI') 
'''
print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs URLLC rates')
ca_params = {'f0': 28e9, 'f1':10e9, 'mode': 2, 'ccRatio': 0.5, 'numEmbbUes':10, 'numUrllcUes':10 }
no_ca_params = {'f0': 28e9, 'mode': 1, 'ccRatio': 0.5, 'ccMan': 2, 'numEmbbUes':10, 'numUrllcUes':10 }

print('Computing stats')
plot_forall_static(paramz_ca=ca_params, param_no_ca=no_ca_params, versus='urllcUdpIPI', fewer_images=True, static='embbUdpIPI')

print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs ccRatio')
ca_params = {'f0': 28e9, 'f1':10e9, 'mode': 2, 'embbUdpIPI': 59}
no_ca_params = {'f0': 28e9, 'mode': 1, 'embbUdpIPI': 59}

print('Computing stats')
plot_forall_static(param_ca=ca_params, param_no_ca=no_ca_params, versus='ccRatio', fewer_images=True, static='urllcUdpIPI')
'''
print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs numEmbbUes')
ca_params = {'f0': 28e9, 'f1':10e9, 'mode': 2, 'embbUdpIPI': 82, 'urllcUdpIPI': 8192, 'ccRatio': 0.5}
no_ca_params = {'f0': 28e9, 'mode': 1, 'embbUdpIPI': 82, 'urllcUdpIPI': 8192, 'ccRatio': 0.5, 'ccMan':0}

print('Computing stats')
plot_forall_static(param_ca=ca_params, param_no_ca=no_ca_params, versus='numEmbbUes', fewer_images=True, static='numUrllcUes')

print('CA using f0=28GHz, f1=10Ghz; non CA using f0=28GhzL: vs numUrllcUes')
ca_params = {'f0': 28e9, 'f1':10e9, 'mode': 2, 'embbUdpIPI': 82, 'urllcUdpIPI': 8192, 'ccRatio': 0.5}
no_ca_params = {'f0': 28e9, 'mode': 1, 'embbUdpIPI': 82, 'urllcUdpIPI': 8192, 'ccRatio': 0.5, 'ccMan':0}

print('Computing stats')
plot_forall_static(param_ca=ca_params, param_no_ca=no_ca_params, versus='numUrllcUes', fewer_images=True, static='numEmbbUes')
