import pyvista as pv
import numpy as np
import plotly.graph_objects as go

def update_plot_pyvista(df, sphere_radius=20):
    # Extract X, Y, and time frame positions
    x_positions = df.iloc[:, 0].to_numpy()
    y_positions = df.iloc[:, 1].to_numpy()
    time_frame = df.iloc[:, 2].to_numpy()

    # Create a PolyData object for the points
    points = pv.PolyData(np.column_stack((x_positions, y_positions, time_frame)))

    # Define a sphere for glyphing
    sphere = pv.Sphere(radius=sphere_radius)

    # Use glyphs to duplicate the sphere at each point's position
    glyphs = points.glyph(scale=False, geom=sphere)

    # Create a plotter and add the glyphs
    plotter = pv.Plotter()
    plotter.add_mesh(glyphs, color="blue")  # Adjust color as needed

    labels = dict(zlabel='Time (frame)', xlabel='X position (nm)', ylabel='Y position (nm)')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.camera_position = 'xy'  # View from top-down perspective
    plotter.show()


def visualize_spatial_clusters_pyvista(all_temporal_clusters, df, sphere_radius=20):
    plotter = pv.Plotter()

    for _, cluster in enumerate(all_temporal_clusters):
        x_coords = []
        y_coords = []
        z_coords = []

        for temporal_cluster in cluster:
            for index, time_frame in temporal_cluster:
                x_coords.append(df.iloc[index, 0])
                y_coords.append(df.iloc[index, 1])
                z_coords.append(time_frame)

        # Create a point cloud
        points = pv.PolyData(np.column_stack((x_coords, y_coords, z_coords)))

        # Create a sphere glyph
        sphere = pv.Sphere(radius=sphere_radius)
        glyphs = points.glyph(scale=False, geom=sphere)

        # Assign a unique color to the glyph
        color = np.random.rand(3)

        # Add the glyph to the plotter
        plotter.add_mesh(glyphs, color=color)

    labels = dict(zlabel='Time (frame)', xlabel='X position (nm)', ylabel='Y position (nm)')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.camera_position = 'xy'  # View from top-down perspective
    plotter.show()

def visualize_temporal_clusters_pyvista(all_temporal_clusters, df, sphere_radius=20):
    plotter = pv.Plotter()

    for _, cluster in enumerate(all_temporal_clusters):

        for temporal_cluster in cluster:
            x_coords = []
            y_coords = []
            z_coords = []
            for index, time_frame in temporal_cluster:
                x_coords.append(df.iloc[index, 0])
                y_coords.append(df.iloc[index, 1])
                z_coords.append(time_frame)

            # Create a point cloud
            points = pv.PolyData(np.column_stack((x_coords, y_coords, z_coords)))

            # Create a sphere glyph
            sphere = pv.Sphere(radius=sphere_radius)
            glyphs = points.glyph(scale=False, geom=sphere)

            # Assign a unique color to the glyph
            color = np.random.rand(3)

            # Add the glyph to the plotter
            plotter.add_mesh(glyphs, color=color)
    
    labels = dict(zlabel='Time (frame)', xlabel='X position (nm)', ylabel='Y position (nm)')
    plotter.show_grid(**labels)
    plotter.add_axes(**labels)
    plotter.camera_position = 'xy'  # View from top-down perspective
    plotter.show()

def plot_2d_points_clusters(all_temporal_clusters, df):
    """Plot 2D scatter plot of localization points with clusters in different colors using plotly."""
    # Create a figure
    fig = go.Figure()

    # Get min and max values for x and y axes
    x_min, x_max = df.iloc[:, 0].min(), df.iloc[:, 0].max()
    y_min, y_max = df.iloc[:, 1].min(), df.iloc[:, 1].max()

    # Count total points to determine performance settings
    total_points = sum(sum(len(temporal_cluster) for temporal_cluster in cluster) 
                      for cluster in all_temporal_clusters)
    
    # Adjust performance settings based on number of points
    is_large_dataset = total_points > 10000
    marker_size = 2 if is_large_dataset else 8
    marker_opacity = 0.6 if is_large_dataset else 1.0
    use_webgl = is_large_dataset

    # Generate a color for each spatial cluster
    colors = [f'rgb({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)})' 
             for _ in range(len(all_temporal_clusters))]

    # Add points for each cluster
    for cluster_idx, cluster in enumerate(all_temporal_clusters):
        # Pre-allocate arrays for better performance
        total_cluster_points = sum(len(temporal_cluster) for temporal_cluster in cluster)
        x_coords = np.empty(total_cluster_points)
        y_coords = np.empty(total_cluster_points)
        
        # Fill arrays more efficiently
        idx = 0
        for temporal_cluster in cluster:
            for index, _ in temporal_cluster:
                x_coords[idx] = df.iloc[index, 0]
                y_coords[idx] = df.iloc[index, 1]
                idx += 1

        # Add scatter plot for this cluster if we have points
        if len(x_coords) > 0:
            cluster_name = "Noise" if cluster_idx == 0 else f'Cluster {cluster_idx}'
            cluster_color = 'grey' if cluster_idx == 0 else colors[cluster_idx]
            
            scatter_kwargs = dict(
                x=x_coords,
                y=y_coords,
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=cluster_color,
                    line=dict(width=0),
                    opacity=marker_opacity
                ),
                name=cluster_name,
                hovertemplate='X: %{x:.1f} nm<br>Y: %{y:.1f} nm<br>Type: %{customdata}<extra></extra>',
                customdata=[cluster_name] * len(x_coords),
                hoverlabel=dict(namelength=0)
            )

            # Use WebGL renderer for large datasets
            if use_webgl:
                fig.add_trace(go.Scattergl(**scatter_kwargs))
            else:
                fig.add_trace(go.Scatter(**scatter_kwargs))

    # Update layout
    fig.update_layout(
        title=f'2D Localization Points with Clusters ({total_points:,} Localizations)',
        xaxis_title='X Position (nm)',
        yaxis_title='Y Position (nm)',
        xaxis=dict(
            range=[x_min, x_max],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            zeroline=False
        ),
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=True,
            gridwidth=1,
            gridcolor='LightGrey',
            zeroline=False
        ),
        showlegend=True,
        template='plotly_white',
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=50),
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        dragmode='zoom',
        plot_bgcolor='white'
    )

    # Update axes to be equal scale
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    # Configure the plot
    config = {
        'displayModeBar': True,
        'responsive': True,
        'scrollZoom': True,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
        'doubleClick': 'reset+autosize',
        'displaylogo': False
    }

    # Show the plot
    fig.show(config=config)
    