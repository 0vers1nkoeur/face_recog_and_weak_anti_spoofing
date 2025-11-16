class ROISet:
    """
    Lightweight read-only mapping describing one or multiple ROIs.
    Supports the + operator for easy combinations, e.g. RPPG.CHEEKS + RPPG.FOREHEAD.
    """
    # Create a list of macros to set the landmark indices for ROI extraction compute_bbox
    RIGHT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148]
    LEFT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377]
    FOREHEAD = [10, 338, 297, 332, 284, 251, 389, 356]

    def __init__(self, mapping):
        """
        Initialize the ROISet with a mapping of ROI names to landmark indices.
        """
        self._mapping = {name: tuple(indices) for name, indices in mapping.items()}

    def items(self):
        """
        Return an iterable view of the ROISet's items (name, indices).
        """
        return self._mapping.items()

    def __iter__(self):
        """
        Return an iterator over the ROISet's keys (ROI names).
        """
        return iter(self._mapping)

    def __len__(self):
        """
        Return the number of ROIs in the ROISet.
        """
        return len(self._mapping)

    def __add__(self, other):
        """
        Combine two ROISet instances or an ROISet with a dict-like object.
        """
        other_mapping = other._mapping if isinstance(other, ROISet) else {
            name: tuple(indices) for name, indices in dict(other).items()
        }
        combined = dict(self._mapping)
        combined.update(other_mapping)
        return ROISet(combined)

    def to_dict(self):
        """
        Convert the ROISet to a standard dictionary.
        """
        return dict(self._mapping)