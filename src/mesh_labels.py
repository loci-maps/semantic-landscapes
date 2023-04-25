import geopandas as gpd
import numpy as np
import pandas as pd
import pyvista as pv


voronoi_file_path =  r"C:\Users\ybinb\VS\pyvista-visualization\assets\voronoi_triangles.geojson"
labels_file_path = r"C:\Users\ybinb\VS\pyvista-visualization\assets\reduced_embeddings_2d.csv"

gdf = gpd.read_file(voronoi_file_path)

reduced_embd = pd.read_csv(labels_file_path)
labels = reduced_embd['filename']

def gdf_to_mesh(gdf, labels):
    vertices = []
    faces = []
    poly_ids = []
    face_labels = []

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

        
        face_label = labels[index]
        face_labels.append(face_label)

    # Combine vertices and face connectivity lists
    vertices = np.vstack(vertices)
    faces = np.hstack([np.hstack([[len(face)], face]) for face in faces])

    mesh = pv.PolyData(vertices, faces)

    labels_encoded = [s.encode('utf-8') for s in face_labels]

    # Add polygon IDs and face labels as scalar arrays
    mesh["PolyIDs"] = np.array(poly_ids) 
    mesh["FaceLabels"] = labels_encoded

    return mesh



mesh = gdf_to_mesh(gdf, labels)

centers = mesh.cell_centers()
print(centers)

# mesh only
mesh.plot(cmap="gist_earth", show_edges=True, show_scalar_bar=True)

# mesh and cell centrs only
pl = pv.Plotter()
pl.add_mesh(mesh, cmap="gist_earth", show_edges=True)
pl.add_points(centers, render_points_as_spheres=True, color='red', point_size=4)
pl.show()

# final result
pl = pv.Plotter()
pl.add_mesh(mesh, cmap="gist_earth", show_edges=True)
pl.add_points(centers, render_points_as_spheres=True, color='red', point_size=1)
pl.add_point_labels(centers, "FaceLabels",  point_size=5, font_size=10)
pl.show()

