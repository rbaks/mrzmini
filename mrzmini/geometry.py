'''
mrzmini.geometry: RotatedBox geometry, adapted from PassportEye's
``passporteye/util/geometry.py``.

Changes vs upstream (to drop heavy dependencies), behaviour preserved:
  * ``sklearn.decomposition.PCA`` -> a small numpy SVD (see :func:`_pca`).
    The resulting box (center/width/height/angle) is invariant to the sign of
    the principal components, so sklearn's ``svd_flip`` convention need not be
    reproduced.
  * ``skimage.transform`` (``rotate`` / ``SimilarityTransform``) ->
    :mod:`mrzmini.imageproc`.
  * The matplotlib ``plot`` method is removed.

Author of original: Konstantin Tretyakov. License: MIT.
'''
import numpy as np

from .imageproc import rotate as _rotate, SimilarityTransform


def _pca(points):
    """Return ``(mean, components)`` matching ``sklearn.decomposition.PCA(2).fit``.

    ``components`` rows are the principal axes ordered by descending variance
    (row 0 = longest axis). ``transform(x) = (x - mean) @ components.T``.
    """
    mean = points.mean(axis=0)
    centered = points - mean
    # full_matrices=False -> Vt has shape (min(n,2), 2); singular values descending.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt
    if components.shape[0] < 2:
        # Degenerate (all points collinear with <2 dims of variance): complete
        # the basis with an orthogonal vector so downstream maths is well-defined.
        v0 = components[0]
        components = np.vstack([v0, [-v0[1], v0[0]]])
    return mean, components


class RotatedBox(object):
    """
    RotatedBox represents a rectangular box centered at (cx,cy) with dimensions width x height,
    rotated by angle radians counterclockwise.

    >>> RotatedBox.from_points([[0,0], [2,1], [0,1], [2,0]])
    RotatedBox(cx=1.0, cy=0.5, width=2.0, height=1.0, angle=0.0)
    """

    def __init__(self, center, width, height, angle, points=None):
        """Creates a new RotatedBox.

        :param points: This parameter may be used to indicate the set of points used to create the box.
        """
        self.center = np.asarray(center, dtype=np.float64)
        self.width = width
        self.height = height
        self.angle = angle
        self.points = points

    def __repr__(self):
        return "RotatedBox(cx={0}, cy={1}, width={2}, height={3}, angle={4})".format(self.cx, self.cy, self.width, self.height, self.angle)

    @property
    def cx(self):
        return self.center[0]

    @property
    def cy(self):
        return self.center[1]

    @property
    def area(self):
        return self.width * self.height

    def approx_equal(self, center, width, height, angle, tol=1e-6):
        "Method mainly useful for testing"
        return abs(self.cx - center[0]) < tol and abs(self.cy - center[1]) < tol and abs(self.width - width) < tol and \
               abs(self.height - height) < tol and abs(self.angle - angle) < tol

    def rotated(self, rotation_center, angle):
        """Returns a RotatedBox that is obtained by rotating this box around a given center by a given angle.

        >>> assert RotatedBox([2, 2], 2, 1, 0.1).rotated([1, 1], np.pi/2).approx_equal([0, 2], 2, 1, np.pi/2+0.1)
        """
        rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        t = np.asarray(rotation_center, dtype=np.float64)
        new_c = np.dot(rot.T, (self.center - t)) + t
        return RotatedBox(new_c, self.width, self.height, (self.angle+angle) % (np.pi*2))

    def as_poly(self, margin_width=0, margin_height=0):
        """Converts this box to a polygon, i.e. 4x2 array, representing the four corners starting from lower left to upper left counterclockwise.

        :param margin_width: The additional "margin" that will be added to the box along its width dimension (from both sides) before conversion.
        :param margin_height: The additional "margin" that will be added to the box along its height dimension (from both sides) before conversion.

        >>> RotatedBox([0, 0], 4, 2, 0).as_poly()
        array([[-2., -1.],
               [ 2., -1.],
               [ 2.,  1.],
               [-2.,  1.]])
        >>> RotatedBox([0, 0], 4, 2, np.pi/2).as_poly()
        array([[ 1., -2.],
               [ 1.,  2.],
               [-1.,  2.],
               [-1., -2.]])
        >>> RotatedBox([0, 0], 0, 0, np.pi/2).as_poly(2, 1)
        array([[ 1., -2.],
               [ 1.,  2.],
               [-1.,  2.],
               [-1., -2.]])
        """
        v_hor = (self.width/2 + margin_width)*np.array([np.cos(self.angle), np.sin(self.angle)])
        v_vert = (self.height/2 + margin_height)*np.array([-np.sin(self.angle), np.cos(self.angle)])
        c = np.array([self.cx, self.cy])
        return np.vstack([c - v_hor - v_vert, c + v_hor - v_vert, c + v_hor + v_vert, c - v_hor + v_vert])

    def extract_from_image(self, img, scale=1.0, margin_width=5, margin_height=5):
        """Extracts the contents of this box from a given image.
        For that the image is "unrotated" by the appropriate angle, and the corresponding part is extracted from it.

        Returns an image with dimensions height*scale x width*scale.
        Note that the box coordinates are interpreted as "image coordinates" (i.e. x is row and y is column),
        and box angle is considered to be relative to the vertical (i.e. np.pi/2 is "normal orientation")

        :param img: a numpy ndarray suitable for image processing.
        :param scale: the RotatedBox is scaled by this value before performing the extraction.
            This is necessary when, for example, the location of a particular feature is determined using a smaller image,
            yet then the corresponding area needs to be extracted from the original, larger image.
            The scale parameter in this case should be width_of_larger_image/width_of_smaller_image.
        :param margin_width: The margin that should be added to the width dimension of the box from each size.
            This value is given wrt actual box dimensions (i.e. not scaled).
        :param margin_height: The margin that should be added to the height dimension of the box from each side.
        :return: a numpy ndarray, corresponding to the extracted region (aligned straight).
        """
        rotate_by = (np.pi/2 - self.angle)*180/np.pi
        img_rotated = _rotate(img, angle=rotate_by, center=[self.center[1]*scale, self.center[0]*scale], resize=True)
        # The resizeable transform will shift the resulting image somewhat wrt original coordinates.
        # When we cut out the box we will compensate for this shift.
        shift_c, shift_r = self._compensate_rotation_shift(img, scale)

        r1 = max(int((self.center[0] - self.height/2 - margin_height)*scale - shift_r), 0)
        r2 = int((self.center[0] + self.height/2 + margin_height)*scale - shift_r)
        c1 = max(int((self.center[1] - self.width/2 - margin_width)*scale - shift_c), 0)
        c2 = int((self.center[1] + self.width/2 + margin_width)*scale - shift_c)
        return img_rotated[r1:r2, c1:c2]

    def _compensate_rotation_shift(self, img, scale):
        """This is an auxiliary method used by extract_from_image.
        It repeats the shift computation performed by skimage's resize-enabled rotate,
        so that the box can be cut out at the right offset.
        """
        ctr = np.asarray([self.center[1]*scale, self.center[0]*scale])
        tform1 = SimilarityTransform(translation=ctr)
        tform2 = SimilarityTransform(rotation=np.pi/2 - self.angle)
        tform3 = SimilarityTransform(translation=-ctr)
        tform = tform3 + tform2 + tform1

        rows, cols = img.shape[0], img.shape[1]
        corners = np.array([
            [0, 0],
            [0, rows - 1],
            [cols - 1, rows - 1],
            [cols - 1, 0]
        ])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()

        # fit output image in new shape
        translation = (minc, minr)
        tform4 = SimilarityTransform(translation=translation)
        tform = tform4 + tform
        tform.params[2] = (0, 0, 1)

        # Compute the shift of the transformed center wrt original
        return (ctr - tform.inverse(ctr)).ravel().tolist()

    @staticmethod
    def from_points(points, box_type='bb'):
        """
        Interpret a given point cloud as a RotatedBox, using PCA to determine the potential orientation (the longest component becomes width)
        This is basically an approximate version of a min-area-rectangle algorithm.

        :param points: An n x 2 numpy array of coordinates.
        :param box_type: The kind of method used to estimate the "box".
            Possible values:
                - `'bb'`, denoting the "bounding box" approach (min/max coordinates of the points correspond to box limits)
                - `'mrz`, denoting a slightly modified technique, suited for MRZ zone detection from contour images.
                          Here the assumption is that the upper and lower bounds of the box are better estimated as the
                          10% and 90% quantile of the corresponding coordinates (rather than 0% and 100%, i.e. min and max).
                          This helps against accidental noise in the contour.
                          The `'mrz'` correction is only applied when there are at least 10 points in the set.
        :returns: a RotatedBox, bounding the given set of points, oriented according to the principal components.

        >>> RotatedBox.from_points([[0,0]])
        RotatedBox(cx=0.0, cy=0.0, width=0.0, height=0.0, angle=0.0)
        >>> assert RotatedBox.from_points([[0,0], [1,1], [2,2]]).approx_equal([1, 1], np.sqrt(8), 0, np.pi/4)
        >>> assert RotatedBox.from_points([[0,0], [1,1], [0,1], [1,0]]).approx_equal([0.5, 0.5], 1, 1, 0.0) # The angle is rather arbitrary here
        >>> assert RotatedBox.from_points([[0,0], [2,1], [0,1], [2,0]]).approx_equal([1, 0.5], 2, 1, 0)
        >>> assert RotatedBox.from_points([[0,0], [2,4], [0,4], [2,0]]).approx_equal([1, 2], 4, 2, np.pi/2)
        >>> assert RotatedBox.from_points([[0,0], [1,1.5], [2,0]]).approx_equal([1, 0.75], 2, 1.5, 0)
        >>> assert RotatedBox.from_points([[0,0], [0,1], [1,1]]).approx_equal([0.25, 0.75], np.sqrt(2), np.sqrt(2)/2, np.pi/4)
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[0] == 1:
            return RotatedBox(points[0], width=0.0, height=0.0, angle=0.0, points=points)

        mean, components = _pca(points)
        # Find the angle
        angle = (np.arctan2(components[0, 1], components[0, 0]) % np.pi)
        if abs(angle - np.pi) < angle:
            # Here the angle is always between -pi and pi
            # If the principal component happened to be oriented so that the angle happens to be > pi/2 by absolute value,
            # we flip the direction
            angle = angle - np.pi if angle > 0 else angle + np.pi
        points_transformed = (points - mean) @ components.T
        ll = np.min(points_transformed, 0)
        ur = np.max(points_transformed, 0)
        wh = ur - ll

        # Now compute and return the bounding box
        if box_type == 'bb' or (box_type == 'mrz' and points.shape[0] < 10):
            # We know that if we rotate the points around mean, we get a box with bounds ur and ll
            # The center of this box is (ur+ll)/2 + mean, which is not the same as the mean,
            # hence to get the center of the original box we need to "unrotate" this box back.
            return RotatedBox(np.dot(components.T, (ll+ur)/2) + mean, width=wh[0], height=wh[1], angle=angle, points=points)
        elif box_type == 'mrz':
            # When working with MRZ detection from contours, we may have minor "bumps" in the contour,
            # that should be ignored at least along the long ("horizontal") side.
            # To do that, we will use 10% and 90% quantiles as the bounds of the box instead of the max and min.
            h_coord = sorted(points_transformed[:, 1])
            n = len(h_coord)
            bottom, top = h_coord[n // 10], h_coord[n * 9 // 10]
            valid_points = np.logical_and(points_transformed[:, 1] >= bottom, points_transformed[:, 1] <= top)
            rb = RotatedBox.from_points(points[valid_points, :], 'bb')
            rb.points = points
            return rb
        else:
            raise ValueError("Unknown parameter value: box_type=%s" % box_type)
