import numpy as np
import matplotlib.pyplot as plt

def get_alt_temp_grids(data, ocean=True):
    asc_longs = np.sort(data['long'].unique())
    desc_lats = np.sort(data['lat'].unique())[::-1]

    temp_grid = []
    alt_grid = []

    for lat in desc_lats:
        temp_row = []
        alt_row = []
        for long in asc_longs:
            point = data.loc[(data['lat'] == lat) & (data['long'] == long)]
            if len(point) == 0 or (point['elevation'].values[0] == 0 and not ocean):
                temp_row.append(np.nan)
                alt_row.append(np.nan)
                continue
            temp_row.append(point['generated_temp'].values[0])
            alt_row.append(point['elevation'].values[0])
        temp_grid.append(temp_row)
        alt_grid.append(alt_row)
    
    return alt_grid, temp_grid

def get_group_avgs(data, xlbls):
    temps = list(data['generated_temp'])
    xlbl_dict = {xlbl:[] for xlbl in set(xlbls)}
    xlbl_avgs = {xlbl:0 for xlbl in set(xlbls)}

    for i, xlbl in enumerate(xlbls):
        xlbl_dict[xlbl].append(temps[i])
    for key, arr in zip(xlbl_dict.keys(), xlbl_dict.values()):
        xlbl_avgs[key] = sum(arr)/len(arr)

    return xlbl_avgs, xlbl_dict

def reconstruct_plot(data, xlbls, plot=True, title='Reconstructed Temps'):
    df_copy = data[['lat', 'long']].copy()

    df_copy['xlbl'] = xlbls
    group_avgs, _ = get_group_avgs(data, xlbls)

    asc_longs = np.sort(df_copy['long'].unique())
    desc_lats = np.sort(df_copy['lat'].unique())[::-1]
    
    grid = []
    points_count = 0

    for lat in desc_lats:
        row = []
        for long in asc_longs:
            point = df_copy[(df_copy['lat'] == lat) & (df_copy['long'] == long)]
            if point.empty:
                row.append(np.nan)
                continue
            
            group = point['xlbl'].values[0]
            row.append(group_avgs[group])
            points_count += 1
        
        grid.append(row)
    
    if plot:
        plt.imshow(grid, cmap='hot_r', interpolation='nearest', vmin=data['generated_temp'].min(), vmax=data['generated_temp'].max())
        plt.colorbar()
        plt.title(title)

    return grid

def reconstruct_contour(data, xlbls, n_clusters, plot=True, title='Reconstructed Contours', grey_back=False):
    df_copy = data[['lat', 'long']].copy()
    df_copy['xlbl'] = xlbls
    group_avgs, _ = get_group_avgs(data, xlbls)

    asc_longs = np.sort(df_copy['long'].unique())
    desc_lats = np.sort(df_copy['lat'].unique())[::-1]

    grid = []
    for lat in desc_lats:
        row = []
        for long in asc_longs:
            point = df_copy[(df_copy['lat'] == lat) & (df_copy['long'] == long)]
            if point.empty:
                row.append(np.nan)
            else:
                group = point['xlbl'].values[0]
                row.append(group_avgs[group])
        grid.append(row)

    grid = np.array(grid)

    if plot:
        nrows, ncols = grid.shape
        x_vals = np.arange(ncols)
        y_vals = np.arange(nrows)
        X, Y = np.meshgrid(x_vals, y_vals)

        plt.figure()
        
        if grey_back:
            nan_mask = np.isnan(grid)
            background = np.zeros((nrows, ncols, 4))
            background[nan_mask, :] = [0.5, 0.5, 0.5, 1.0]
            background[~nan_mask, :] = [1.0, 1.0, 1.0, 0.0]
            plt.imshow(background, origin='upper', aspect='equal')

        plt.contour(X, Y, grid, levels=n_clusters, colors='black', linewidths=0.5)
        plt.gca().set_aspect('equal')

        plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
        plt.locator_params(axis='x', nbins=4)
        plt.xlabel('Longitude', fontsize=13)
        plt.ylabel('Latitude', fontsize=13)

        if title:
            plt.title(title, fontsize=13)
        plt.tight_layout()
        plt.show()

    return grid

def by_cluster_err(train, xlbls, test, test_xlbls, err='sq'):
    group_avgs, _ = get_group_avgs(train, xlbls)
    _, test_groups = get_group_avgs(test, test_xlbls)
    abs_err = 0

    for group in test_groups.keys():
        true_temps = np.array(test_groups[group])

        pred_temp = group_avgs[group]

        if err == 'sq':
            abs_err += np.mean((true_temps - pred_temp)**2)
        elif err == 'abs':
            abs_err += np.mean(np.abs(true_temps - pred_temp))
    
    return abs_err / len(test_groups.keys())

def by_point_err(train, xlbls, test, test_xlbls, err='sq'):
    group_avgs, _ = get_group_avgs(train, xlbls)
    _, test_groups = get_group_avgs(test, test_xlbls)
    abs_err = 0

    for group in test_groups.keys():
        true_temps = np.array(test_groups[group])

        pred_temp = group_avgs[group]

        if err == 'sq':
            abs_err += np.sum((true_temps - pred_temp)**2)
        elif err == 'abs':
            abs_err += np.sum(np.abs(true_temps - pred_temp))
            
    return abs_err / len(test)

def plot_pred_distribution(train_data, xlbls, resolution=None, n_clusters=None, yminmax=None):
    resolution = "unknown resolution" if not resolution else resolution
    n_clusters = "unknown" if not n_clusters else n_clusters

    len_data = len(train_data)

    avgs, lbl_dict = get_group_avgs(train_data, xlbls)
    avgs_list = [(key, avgs[key]) for key in avgs.keys()]
    avgs_list = sorted(avgs_list, key=lambda x: x[1], reverse=True)
    x_axis = np.linspace(0, len_data - 1, len_data)

    temp_preds = []
    for clus, avg in avgs_list:
        for _ in range(len(lbl_dict[clus])):
            temp_preds.append(avg)
    temp_preds = np.array(temp_preds).flatten()

    _ = plt.figure(figsize=(12, 6))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    print(f'MIN TEMP FOR THIS DATASET: {train_data["generated_temp"].min()}')

    line_color = 'red'
    ax1.set_xlabel('Index of Points Sorted by Cluster')
    ax1.set_ylabel('Average Temperature', color=line_color)
    ax1.plot(x_axis, temp_preds, color=line_color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=line_color)
    if yminmax:
        ax1.set_ylim(*yminmax)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title(f'Average Temp and Count of Points by Cluster - {resolution} + {n_clusters} clusters', fontsize=14)
    plt.show()

def gen_elevation_grads(data, d, nan_val=-100, white_back=False, grey_back=False):
    grid = np.array(data)

    rows, cols = grid.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    U, V = np.zeros_like(grid, dtype=float), np.zeros_like(grid, dtype=float)

    # Compute gradients toward the highest elevation within `d` range
    for i in range(rows):
        for j in range(cols):
            if np.isnan(grid[i, j]):
                U[i, j] = np.nan
                V[i, j] = np.nan
                continue

            max_elevation = grid[i, j]
            best_dx, best_dy = 0, 0

            # Search within depth `d`
            for dx in range(-d, d + 1):
                for dy in range(-d, d + 1):
                    ni, nj = i + dx, j + dy
                    if 0 <= ni < rows and 0 <= nj < cols and (dx != 0 or dy != 0):
                        if grid[ni, nj] > max_elevation:
                            max_elevation = grid[ni, nj]
                            best_dx, best_dy = dx, dy

            U[i, j] = best_dy
            V[i, j] = best_dx

    mag = np.sqrt(U**2 + V**2)
    norm_U, norm_V = U / (mag + 1e-10), V / (mag + 1e-10)  # Avoid division by zero

    # Plot the quiver plot
    fig, ax = plt.subplots(figsize=(5, 6), dpi=150)
    ax.set_title(f'Gradient Toward Highest Elevation (Search Depth={d})', fontsize=10)
    ax.quiver(X, Y, norm_U, -norm_V, color='black', pivot='mid', scale=40)  # Flip V for correct image alignment

    grid = np.ma.masked_invalid(grid)
    if not white_back:
        cmap = plt.get_cmap('terrain')
        if grey_back:
            cmap.set_bad('grey')
        cbar = fig.colorbar(ax.imshow(grid, cmap=cmap, origin='upper'), ax=ax)
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Elevation (m)', fontsize=13)
    else:
        cmap = plt.get_cmap('gray_r')
        if grey_back:
            cmap.set_bad('grey')
        ax.imshow(np.zeros_like(grid), cmap=cmap, origin='upper')

    ax.set_xticks([])
    ax.set_yticks([])

    return U, V

def grad_angles(U, V, title='Angle Representation of Gradients', grey_back=False):
    angles_raw = np.arctan2(-V, U)
    angles_raw[angles_raw < 0] += 2 * np.pi
    angles = angles_raw * 180 / np.pi  # Convert to degrees
    angles = np.ma.masked_invalid(angles)

    fig, ax = plt.subplots(figsize=(5, 6), dpi=150)
    ax.set_title(title, fontsize=10)
    cmap = plt.get_cmap('twilight_shifted')
    if grey_back:
        cmap.set_bad('grey')

    im = ax.imshow(angles, cmap=cmap, origin='upper')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Gradient Direction (Degrees)', fontsize=13)

    ax.set_xticks([])
    ax.set_yticks([])

    return angles

def plot_area(data, units, title='Area', grey_back=False, color=None):
    if units in ['elevation', 'alt', 'altitude']:
        label = 'Elevation (m)'
        cmap = plt.get_cmap('terrain')
    elif units in ['temp', 'temperature']:
        label = 'Temperature (°C)'
        cmap = plt.get_cmap('hot_r')
    if color:
        cmap = plt.get_cmap(color)
    if grey_back:
        cmap.set_bad('grey')

    plt.imshow(data, cmap=cmap, interpolation='nearest')

    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)
    plt.locator_params(axis='x', nbins=4)

    plt.xlabel('Longitude', fontsize=13)
    plt.ylabel('Latitude', fontsize=13)
    cbar = plt.colorbar()
    cbar.set_ticks([])
    cbar.set_label(label, fontsize=12)
    if title:
        plt.title(title, fontsize=13)
    plt.show()
