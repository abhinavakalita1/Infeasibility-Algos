# Infeasibility-Algos
<u>generate_pointcloud, view_clusters :</u> generates pointcloud from obstacles, uses alpha reconstruction and saves the meshes as obj files <br>
cspace_scattering: Rays scatter from start/goal only, zero sdf points are found on rays using regula falsi. midpoints are calculated<br>
cspace_3d_frontier: rays are scattered from points inside -ve sdf region to create new points. they are joined to form graph structure. closedness cannot be interpreted directly<br>
cspace_3d_hexagon: uses rays from start/goal to create Hexagons. WORKS!<br>
cspace_qhull: uses cspace_3d_frontier but then applies QHULL over it. traces all tets on the surface of QHULL to get +ve and -ve sdf regions. WORKS!!<br>
cspace_3d_boundary: Original algorithm, boudnary tracing. can be used as ground truth<br>
ground_truth_3d: absolute ground truth.<br>
