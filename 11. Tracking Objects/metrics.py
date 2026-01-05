def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """

    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])
    
    intersection_width = x_right - x_left
    intersection_height = y_bottom - y_top
    
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    intersection_area = intersection_width * intersection_height
    union_area = bbox1_area + bbox2_area - intersection_area
    
    return intersection_area / union_area


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        # Write code here
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {det[0]: det[1:] for det in frame_obj}
        hyp_dict = {hyp[0]: hyp[1:] for hyp in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        id_matches = {}

        for obj_id, hyp_id in id_matches.items():
            if obj_id in obj_dict and hyp_id in hyp_dict:
                curr_iou = iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                if curr_iou > threshold:
                    dist_sum += curr_iou
                    match_count += 1
                    id_matches[obj_id] = hyp_id
                    del obj_dict[obj_id]
                    del hyp_dict[hyp_id]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        curr_matches = []

        for obj_id, obj_bbox in obj_dict.items():
            for hyp_id, hyp_bbox in hyp_dict.items():
                curr_iou = iou_score(obj_bbox, hyp_bbox)
                if curr_iou > threshold:
                    curr_matches.append((curr_iou, obj_id, hyp_id))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        curr_matches.sort(reverse=True, key=lambda x: x[0])
        used_obj = set()
        used_hyp = set()

        for iou, obj_id, hyp_id in curr_matches:
            if obj_id not in used_obj and hyp_id not in used_hyp:
                dist_sum += iou
                match_count += 1
                id_matches[obj_id] = hyp_id
                used_obj.add(obj_id)
                used_hyp.add(hyp_id)

                if obj_id in obj_dict:
                    del obj_dict[obj_id]
                if hyp_id in hyp_dict:
                    del hyp_dict[hyp_id]
        

        # Step 5: Update matches with current matched IDs
        matches = id_matches

    # Step 6: Calculate MOTP
    MOTP = dist_sum / match_count

    return MOTP


def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0
    total_objects = 0

    matches = {}  # matches between object IDs and hypothesis IDs

    # For every frame
    for frame_obj, frame_hyp in zip(obj, hyp):
        total_objects += len(frame_obj)
        # Step 1: Convert frame detections to dict with IDs as keys
        obj_dict = {det[0]: det[1:] for det in frame_obj}
        hyp_dict = {hyp[0]: hyp[1:] for hyp in frame_hyp}

        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        id_matches = {}

        for obj_id, hyp_id in list(matches.items()):
            if obj_id in obj_dict and hyp_id in hyp_dict:
                iou = iou_score(obj_dict[obj_id], hyp_dict[hyp_id])
                if iou > threshold:
                    id_matches[obj_id] = hyp_id
                    dist_sum += iou
                    match_count += 1
                    del obj_dict[obj_id]
                    del hyp_dict[hyp_id]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        curr_matches = []
        for obj_id, obj_bbox in obj_dict.items():
            for hyp_id, hyp_bbox in hyp_dict.items():
                iou = iou_score(obj_bbox, hyp_bbox)
                if iou > threshold:
                    curr_matches.append((iou, obj_id, hyp_id))

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        curr_matches.sort(key=lambda x: x[0], reverse=True)

        for iou, obj_id, hyp_id in curr_matches:
            if obj_id in obj_dict and hyp_id in hyp_dict:

        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error        
                if obj_id in matches and matches[obj_id] != hyp_id:
                    mismatch_error += 1

                id_matches[obj_id] = hyp_id
                dist_sum += iou
                match_count += 1
                del obj_dict[obj_id]
                del hyp_dict[hyp_id]

        # Step 6: Update matches with current matched IDs
        matches.update(id_matches)

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        false_positive += len(hyp_dict)
        # All remaining objects are considered misses
        missed_count += len(obj_dict)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count
    MOTA = 1 - ((missed_count + false_positive + mismatch_error) / total_objects)

    return MOTP, MOTA
