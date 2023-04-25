import geopandas as gpd
import numpy as np
import pyvista as pv


file_path = r"C:\Users\ybinb\VS\pyvista-visualization\assets\nested_voronoi_triangles.geojson"

file_path_1 = r"C:\Users\ybinb\VS\pyvista-visualization\assets\voronoi_split_1.geojson"
file_path_2 = r"C:\Users\ybinb\VS\pyvista-visualization\assets\voronoi_split_2.geojson"
file_path_3 = r"C:\Users\ybinb\VS\pyvista-visualization\assets\voronoi_split_3.geojson"



# the complete nested voronoi diagram
gdf = gpd.read_file(file_path)

# voronoi diagram splits
# split_1_gdf = gpd.read_file(file_path_1)
# split_2_gdf = gpd.read_file(file_path_2)
# split_3_gdf = gpd.read_file(file_path_3)


def gdf_to_mesh(gdf):
    
    vertices = []
    faces = []
    poly_ids = []

    for index, row in gdf.iterrows():
        polygon = row['geometry']
        if not polygon.is_valid:
            continue

        exterior = np.array(polygon.exterior.coords)
        exterior_3d = np.hstack([exterior, np.zeros((len(exterior), 1))])  # Add z-coordinate of zero
        vertices.append(exterior_3d)

        face = np.arange(len(exterior), dtype=np.int64) + len(np.vstack(vertices)[:-len(exterior)])
        faces.append(face)

        # Assign an ID to each polygon
        poly_ids.extend([index] * len(face))
        

    # Combine vertices and face connectivity lists
    vertices = np.vstack(vertices)
    faces = np.hstack([np.hstack([[len(face)], face]) for face in faces])

    mesh = pv.PolyData(vertices, faces)

    
    mesh["PolyIDs"] = np.array(poly_ids) 

    return mesh





voronoi_mesh = gdf_to_mesh(gdf)


warp = voronoi_mesh.warp_by_scalar(factor=0.0001) #0.5e-5
surf = warp.delaunay_2d(alpha=0.5)
smooth = surf.smooth(n_iter=150)


voronoi_mesh.plot(cmap="gist_earth", show_edges=True, show_scalar_bar=True)
warp.plot(cmap="gist_earth", show_edges=True, show_scalar_bar=True)
surf.plot(cmap="gist_earth", show_edges=False, show_scalar_bar=True)
smooth.plot(cmap="gist_earth", show_edges=False, show_scalar_bar=True)


# mesh_1 = gdf_to_mesh(split_1_gdf)
# mesh_2 = gdf_to_mesh(split_2_gdf)
# mesh_3 = gdf_to_mesh(split_3_gdf)

# # Splits aligend side by side test
# #--------------------------------------------------------------------
# plotter = pv.Plotter()
# plotter.add_mesh(mesh_1, cmap="gist_earth", show_edges=False)
# plotter.add_mesh(mesh_2, cmap="gist_earth", show_edges=False)
# plotter.add_mesh(mesh_3, cmap="gist_earth", show_edges=False)
# plotter.show()

#  #Splits translated test
# #--------------------------------------------------------------------
# split_1_translated = mesh_1.translate((0, 0, -0.05), inplace=False)
# split_3_translated = mesh_3.translate((0, 0, 0.06), inplace=False)

# plotter = pv.Plotter()
# plotter.add_mesh(split_1_translated, cmap="gist_earth", show_edges=False)
# plotter.add_mesh(mesh_2, cmap="gist_earth", show_edges=False)
# plotter.add_mesh(split_3_translated, cmap="gist_earth", show_edges=False)
# plotter.show()


