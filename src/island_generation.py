import numpy as np
import pandas as pd

import geopandas as gpd
from shapely import MultiPoint, voronoi_polygons, convex_hull
from scipy.interpolate import griddata

import pyvista as pv
from pyvista import examples


embeddings_npz = np.load('combined_reduced_embeddings.npz')
filenames = pd.read_csv('combined_filenames.csv', header=None)


# [wrap factor, aplha] for flat_mesh_to_terrain func
dim_reduction_values = {
    "pca5": [0.00009, 4.5],
    "umap2": [0.0009, 3.5],
    "tsne2": [0.0045, 9.5],
    "umap5": [0.00005, 4.5]
}

def create_voronoi_gdf(polygons, edges):
    voronoi_polygons = []

    for i, polygon in enumerate(polygons.geoms):
        clipped_polygon = polygon.intersection(edges)
        voronoi_polygons.append(clipped_polygon)

    gdf = gpd.GeoDataFrame(geometry=voronoi_polygons)
    return gdf

def gdf_to_flat_mesh(gdf, labels, rgb):
    
    gdf['area'] = gdf['geometry'].area
    gdf = gdf.sort_values('area').reset_index(drop=True)

    vertices = []
    faces = []
    
    poly_ids = []
    face_labels = []
    face_colors = []

    for index, row in gdf.iterrows():
        polygon = row['geometry']
        if not polygon.is_valid:
            continue

        exterior = np.array(polygon.exterior.coords)
        exterior_3d = np.hstack([exterior, np.zeros((len(exterior), 1))]) 
        vertices.append(exterior_3d)

        face = np.arange(len(exterior), dtype=np.int64) + len(np.vstack(vertices)[:-len(exterior)])
        faces.append(face)

        # Assign an ID to each polygon
        poly_ids.extend([index] * len(face))

        
        face_label = labels[index]
        face_labels.append(face_label)

        face_color = rgb[index]
        face_colors.append(face_color)

    vertices = np.vstack(vertices)
    faces = np.hstack([np.hstack([[len(face)], face]) for face in faces])

    mesh = pv.PolyData(vertices, faces)
   
    labels_encoded = [s.encode('utf-8') for s in face_labels]

    
    mesh["PolyIDs"] = -np.array(poly_ids) 
    mesh["FaceLabels"] = labels_encoded
    mesh["FaceColors"] = face_colors
    

    return mesh

def flat_mesh_to_terrain(mesh, dim_reduction='umap5'):
    if dim_reduction not in dim_reduction_values:
        raise ValueError(f"Unknown dim_reduction value: {dim_reduction}")
        
    factor, alpha = dim_reduction_values[dim_reduction]
    
    warp = mesh.warp_by_scalar(factor=factor)
    surf = warp.delaunay_2d(alpha=alpha)
    surf_smoothed = surf.smooth(n_iter=150)
    
    return surf_smoothed

def interpolate_z(file_points, terrain):
    file_points_coords = file_points.points
    terrain_coords = terrain.points

    # Use only X and Y for interpolation
    points_xy = file_points_coords[:, :2]
    terrain_xy = terrain_coords[:, :2]

    # Interpolate Z values from terrain onto point positions
    interpolated_z = griddata(terrain_xy, terrain_coords[:, 2], points_xy)

    # Create a new set of points with the interpolated Z values
    new_points = np.hstack((points_xy, interpolated_z.reshape(-1, 1)))

    file_points.points = new_points
    return file_points

def file_coord_to_dataframe(points, mesh, csv_file='test_dataframe.csv'):
    
    file_coords = points.points
    face_labels = mesh.cell_data['FaceLabels']

    df = pd.DataFrame(file_coords, columns=['X', 'Y', 'Z'])
    df['FaceLabels'] = face_labels

    df.to_csv(csv_file, index=False)

#---------------------------

# ['pca5', 'tsne2', 'umap5', 'umap2']
rgb = embeddings_npz['pca5'][:, :3]
xy = embeddings_npz['pca5'] [:,:2]

points = MultiPoint(xy)
hull = convex_hull(points) 
polygons = voronoi_polygons(points)

gdf = create_voronoi_gdf(polygons, hull)

mesh = gdf_to_flat_mesh(gdf, filenames[0], rgb)
centers = mesh.cell_centers()

smooth = flat_mesh_to_terrain(mesh, 'pca5')

image_path = examples.planets.download_milkyway_sky_background(load=False)


file_points = interpolate_z(centers, smooth)


pl = pv.Plotter()
pl.add_mesh(smooth, cmap="gist_earth", scalars='PolyIDs')
pl.add_point_labels(file_points, "FaceLabels",  point_size=5, font_size=10)
#pl.add_background_image(image_path)
pl.show()
