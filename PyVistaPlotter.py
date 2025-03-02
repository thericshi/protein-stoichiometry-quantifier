import pyvista as pv
import numpy as np

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
    # light = pv.Light(position=(10, 10, 10))
    # light.diffuse_color = 1.0, 0.0, 0.0
    # plotter.add_light(light)
    # plotter.enable_ssao()
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
    