import geopandas as gpd
import numpy as np
import pyvista as pv


file_path = r"C:\Users\ybinb\VS\pyvista-visualization\assets\triangles.geojson"

gdf = gpd.read_file(file_path)

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

    # Add polygon IDs as a scalar array
    mesh["PolyIDs"] = np.array(poly_ids) 

    return mesh


mesh = gdf_to_mesh(gdf)

warp = mesh.warp_by_scalar(factor=0.0001) #0.5e-5
surf = warp.delaunay_2d(alpha=0.5)
smooth = surf.smooth(n_iter=150)


mesh.plot(cmap="gist_earth", show_edges=True, show_scalar_bar=True)
warp.plot(cmap="gist_earth", show_scalar_bar=True)
surf.plot(cmap="gist_earth")
smooth.plot(cmap="gist_earth", show_edges=False)
smooth.plot(cmap="gist_earth", show_edges=True)

