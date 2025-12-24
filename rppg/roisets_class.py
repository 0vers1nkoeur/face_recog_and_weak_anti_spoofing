class ROISet:
    """
    Lightweight read-only mapping describing one or multiple ROIs.
    Supports the + operator for easy combinations, e.g. RPPG.CHEEKS + RPPG.FOREHEAD.
    """
    # Create a list of macros to set the landmark indices for ROI extraction compute_bbox
    RIGHT_CHEEK = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148] #DO NOT USE ANYMORE
    NEW_RIGHT_CHEEK = [129, 126, 100, 119, 118, 117, 123, 147, 213, 192, 135, 210, 212, 207, 206, 203]
    LEFT_CHEEK = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377] #DO NOT USE ANYMORE
    NEW_LEFT_CHEEK = [358, 355, 329, 348, 347, 346, 352, 376, 433, 416, 364, 394, 431, 430, 432, 436, 426, 423]
    FOREHEAD = [9, 107, 65, 105, 104, 103, 67, 109, 10, 338, 297, 332, 333, 334, 296, 336]

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