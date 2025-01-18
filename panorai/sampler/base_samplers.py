from abc import ABC, abstractmethod
import numpy as np


class Sampler(ABC):
    """Abstract base class for sphere samplers."""
    @abstractmethod
    def get_tangent_points(self):
        """
        Generate tangent points (latitude, longitude) on the sphere.
        """
        pass

    def update(self, **kwargs):
        """Update the sampler with new parameters."""
        self.params.update(kwargs)
        
class CubeSampler(Sampler):
    """Generates tangent points for a cube-based projection."""
    def __init__(self, **kwargs):
        """
        Initialize the CubeSampler with optional parameters from kwargs.

        :param kwargs: Additional parameters (unused but accepted for compatibility).
        """
        self.params = kwargs
        pass  # CubeSampler doesn't require specific parameters.

    def get_tangent_points(self):
        """
        Returns tangent points for cube faces (latitude, longitude).
        """
        return [
            (0, 0),     # Front
            (0, 90),    # Right
            (0, 180),   # Back
            (0, -90),   # Left
            (90, 0),    # Top
            (-90, 0)    # Bottom
        ]


class IcosahedronSampler(Sampler):
    """Generates tangent points for an icosahedron-based projection."""
    def __init__(self, **kwargs):
        """
        Initialize the IcosahedronSampler with parameters from kwargs.

        :param kwargs: Additional parameters. Expected 'subdivisions' key for subdivisions.
        """
        self.params = kwargs


    def _generate_icosahedron(self):
        """
        Generate vertices and faces of the icosahedron with subdivisions.
        """
        subdivisions = self.params.get('subdivisions', 0)  # Default to 0 subdivisions.
        
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        verts = [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
        ]
        verts = [self._normalize_vertex(*v) for v in verts]

        faces = [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [5, 4, 9], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
        ]

        for _ in range(subdivisions):
            mid_cache = {}
            faces_subdiv = []
            for tri in faces:
                v1 = self._midpoint(verts, mid_cache, tri[0], tri[1])
                v2 = self._midpoint(verts, mid_cache, tri[1], tri[2])
                v3 = self._midpoint(verts, mid_cache, tri[2], tri[0])
                faces_subdiv.extend([
                    [tri[0], v1, v3],
                    [tri[1], v2, v1],
                    [tri[2], v3, v2],
                    [v1, v2, v3]
                ])
            faces = faces_subdiv

        return np.array(verts), faces

    @staticmethod
    def _normalize_vertex(x, y, z):
        """
        Normalize a vertex to the unit sphere.
        """
        length = np.sqrt(x**2 + y**2 + z**2)
        return [i / length for i in (x, y, z)]

    @staticmethod
    def _midpoint(verts, cache, p1, p2):
        """
        Find or create the midpoint between two vertices.
        """
        smaller, larger = sorted([p1, p2])
        key = (smaller, larger)
        if key in cache:
            return cache[key]

        v1 = verts[p1]
        v2 = verts[p2]
        mid = [(v1[i] + v2[i]) / 2 for i in range(3)]
        mid_normalized = IcosahedronSampler._normalize_vertex(*mid)
        verts.append(mid_normalized)
        cache[key] = len(verts) - 1
        return cache[key]

    def get_tangent_points(self):
        """
        Compute tangent points from the face centers.
        """
        vertices, faces = self._generate_icosahedron()
        face_centers = np.mean(vertices[np.array(faces)], axis=1)
        return [self._cartesian_to_lat_lon(center) for center in face_centers]

    @staticmethod
    def _cartesian_to_lat_lon(cartesian):
        """
        Convert Cartesian coordinates to latitude and longitude.
        """
        x, y, z = cartesian
        latitude = np.degrees(np.arcsin(z))
        longitude = np.degrees(np.arctan2(y, x))
        return latitude, longitude


class FibonacciSampler(Sampler):
    """Generates tangent points using the Fibonacci sphere method."""
    def __init__(self, **kwargs):
        """
        Initialize the FibonacciSampler with parameters from kwargs.

        :param kwargs: Additional parameters. Expected 'n_points' key for number of points.
        """
        self.params = kwargs


    def get_tangent_points(self):
        """
        Generate tangent points using Fibonacci sphere sampling.
        """
        n_points = self.params.get('n_points', 10) 
        indices = np.arange(0, self.n_points) + 0.5
        phi = 2 * np.pi * indices / ((1 + np.sqrt(5)) / 2)  # Golden angle
        theta = np.arccos(1 - 2 * indices / n_points)  # Polar angle
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return [self._cartesian_to_lat_lon((x[i], y[i], z[i])) for i in range(len(x))]

    @staticmethod
    def _cartesian_to_lat_lon(cartesian):
        """
        Convert Cartesian coordinates to latitude and longitude.
        """
        x, y, z = cartesian
        latitude = np.degrees(np.arcsin(z))
        longitude = np.degrees(np.arctan2(y, x))
        return latitude, longitude
    

SAMPLER_CLASSES = {
    "CubeSampler": CubeSampler,
    "IcosahedronSampler": IcosahedronSampler,
    "FibonacciSampler": FibonacciSampler,
    # Add other sampler classes here if needed
}

