from typing import Dict, List, Optional, Union

import numpy as np

import mmcv
from mmengine.dist import master_only
from mmdet.visualization import DetLocalVisualizer
from mmengine.structures import InstanceData
from mmdet.registry import VISUALIZERS
from mmdet.visualization.palette import get_palette
from mmdet.structures import DetDataSample


@VISUALIZERS.register_module()
class OpenVocPoseVisualizer(DetLocalVisualizer):
    """MMDetection Local Visualizer.

    Args:
        name (str): Name of the instance. Defaults to 'visualizer'.
        image (np.ndarray, optional): the origin image to draw. The format
            should be RGB. Defaults to None.
        vis_backends (list, optional): Visual backend config list.
            Defaults to None.
        save_dir (str, optional): Save file dir for all storage backends.
            If it is None, the backend storage will not save any data.
        line_width (int, float): The linewidth of lines.
            Defaults to 3.
        alpha (int, float): The transparency of bboxes or mask.
            Defaults to 0.8.
    """

    def __init__(self,
                 name: str = 'visualizer',
                 image: Optional[np.ndarray] = None,
                 vis_backends: Optional[Dict] = None,
                 save_dir: Optional[str] = None,
                 line_width: Union[int, float] = 3,
                 alpha: float = 0.8) -> None:
        super().__init__(
            name=name,
            image=image,
            vis_backends=vis_backends,
            save_dir=save_dir,
            line_width=line_width,
            alpha=alpha)

    def _draw_instances(self, image: np.ndarray, instances: List[InstanceData],
                        classes: Optional[List[str]],
                        palette: Optional[List[tuple]],
                        pred_relation_score_thr: float = 0.3) -> np.ndarray:
        """Draw instances of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            instances (:obj:`InstanceData`): Data structure for
                instance-level annotations or predictions.
            classes (List[str], optional): Category information.
            palette (List[tuple], optional): Palette information
                corresponding to the category.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        self.set_image(image)
        
        # Draw keypoints and connections if they exist
        if 'keypoints' in instances and len(instances.keypoints) > 0:
            keypoints = instances.keypoints
            keypoint_labels = instances.get('labels', None)
            keypoint_scores = instances.get('scores', None)
            keypoint_relation_scores = instances.get('relation_scores', None)

            # Get colors for keypoints based on their labels
            max_kp_label = 0
            if keypoint_labels is not None:
                max_kp_label = int(max(keypoint_labels) if len(keypoint_labels) > 0 else 0)
            num_relations = keypoint_relation_scores.shape[-1] if keypoint_relation_scores is not None else 0
            kp_palette = get_palette(palette, max_kp_label + 1 + num_relations)

            # Draw edges between keypoints if relation scores exist
            if keypoint_relation_scores is not None:
                
                for i in range(keypoint_relation_scores.shape[0]):
                    for j in range(keypoint_relation_scores.shape[1]):
                        for k in range(keypoint_relation_scores.shape[2]):
                            relation_score = keypoint_relation_scores[i, j, k]
                            if relation_score > pred_relation_score_thr:

                                kp1 = keypoints[i][:2]  # x, y coordinates
                                kp2 = keypoints[j][:2]
                                
                                self.draw_lines(
                                    x_datas=np.array([kp1[0], kp2[0]]),
                                    y_datas=np.array([kp1[1], kp2[1]]),
                                    colors=kp_palette[max_kp_label + k],
                                    line_styles='-',
                                    line_widths=1.5
                                )
            
            # Draw keypoints
            for i, kp in enumerate(keypoints):
                x, y = kp[:2]
                
                # Skip keypoints with visibility flag set to 0 if present
                if len(kp) > 2 and kp[2] == 0:
                    continue
                
                # Determine keypoint color based on label
                color = 'red'  # Default color
                if keypoint_labels is not None:
                    color = kp_palette[keypoint_labels[i]]
                
                # Draw keypoint
                self.draw_circles(
                    np.array([[x, y]]),
                    radius=np.array([5]),  
                    face_colors=color,
                    edge_colors='black',
                    alpha=0.8,
                    line_widths=1.5
                )
                
                # Add keypoint label/score text
                label_text = ""
                if hasattr(instances, 'label_names') and i < len(instances.label_names):
                    label_text = instances.label_names[i]
                elif keypoint_labels is not None:
                    label_text = f"class {keypoint_labels[i]}"
                
                if keypoint_scores is not None and i < len(keypoint_scores):
                    score_text = f"{keypoint_scores[i]:.2f}"
                    label_text = f"{label_text}: {score_text}" if label_text else score_text
                
                if label_text:
                    self.draw_texts(
                        label_text,
                        np.array([[x + 5, y + 5]]),  # Convert tuple to numpy array
                        colors=color,
                        font_sizes=8,
                        bboxes=[{
                            'facecolor': 'white',
                            'alpha': 0.7,
                            'pad': 0.2,
                            'edgecolor': 'none'
                        }]
                    )

        # TODO: draw legend

        return self.get_image()
    

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            data_sample: Optional['DetDataSample'] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            out_file: Optional[str] = None,
            pred_score_thr: float = 0.3,
            pred_relation_score_thr: float = 0.3,
            step: int = 0) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. t is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            data_sample (:obj:`DetDataSample`, optional): A data
                sample that contain annotations and predictions.
                Defaults to None.
            draw_gt (bool): Whether to draw GT DetDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction DetDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            pred_score_thr (float): The threshold to visualize the bboxes
                and masks. Defaults to 0.3.
            pred_relation_score_thr (float): The threshold to visualize the
                relations. Defaults to 0.3.
            step (int): Global step value to record. Defaults to 0.
        """
        image = image.clip(0, 255).astype(np.uint8)
        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)

        gt_img_data = None
        pred_img_data = None

        if data_sample is not None:
            data_sample = data_sample.cpu()

        if draw_gt and data_sample is not None:
            gt_img_data = image
            if 'gt_instances' in data_sample:
                gt_img_data = self._draw_instances(image,
                                                   data_sample.gt_instances,
                                                   classes, palette)

        if draw_pred and data_sample is not None:
            pred_img_data = image
            if 'pred_instances' in data_sample:
                pred_instances = data_sample.pred_instances

                pred_indices = np.where(pred_instances.scores > pred_score_thr)[0]
                if len(pred_indices) > 0:
                    pred_instances = pred_instances[pred_indices]

                    if 'relation_scores' in pred_instances:
                        pred_instances.relation_scores = pred_instances.relation_scores[:][:, pred_indices]

                pred_img_data = self._draw_instances(image, pred_instances,
                                                     classes, palette, pred_relation_score_thr=pred_relation_score_thr)

        if gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        elif pred_img_data is not None:
            drawn_img = pred_img_data
        else:
            # Display the original image directly if nothing is drawn.
            drawn_img = image

        # It is convenient for users to obtain the drawn image.
        # For example, the user wants to obtain the drawn image and
        # save it as a video during video inference.
        self.set_image(drawn_img)

        if show:
            self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            mmcv.imwrite(drawn_img[..., ::-1], out_file)
        else:
            self.add_image(name, drawn_img, step)