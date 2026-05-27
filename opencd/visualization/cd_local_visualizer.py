from typing import Optional, Sequence

import cv2
import mmcv
import numpy as np
from mmengine.structures import PixelData
from mmengine.dist import master_only

from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from opencd.registry import VISUALIZERS


@VISUALIZERS.register_module()
class CDLocalVisualizer(SegLocalVisualizer):
    """Change Detection Local Visualizer. """

    def _draw_mask(self, mask: PixelData, classes,
                   palette, with_labels: bool) -> np.ndarray:
        mask_h, mask_w = mask.data.shape[-2:]
        mask_img = np.zeros((mask_h, mask_w, 3), dtype=np.uint8)
        return self._draw_sem_seg(mask_img, mask, classes, palette,
                                  with_labels)

    def _resize_like(self, image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if image.shape[:2] == reference.shape[:2]:
            return image
        return mmcv.imresize(image, (reference.shape[1], reference.shape[0]))

    def _make_four_panel(self,
                         panels: Sequence[np.ndarray],
                         titles: Sequence[str],
                         gap: int = 8,
                         title_height: int = 56) -> np.ndarray:
        height, width = panels[0].shape[:2]
        canvas_width = width * len(panels) + gap * (len(panels) - 1)
        canvas_height = height + title_height
        canvas = np.full((canvas_height, canvas_width, 3), 18, dtype=np.uint8)

        for idx, (panel, title) in enumerate(zip(panels, titles)):
            x = idx * (width + gap)
            canvas[title_height:title_height + height, x:x + width] = panel
            text_size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                           1.25, 3)
            text_x = x + max((width - text_size[0]) // 2, 0)
            text_y = 38
            cv2.putText(canvas, title, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.25, (240, 240, 240), 3,
                        cv2.LINE_AA)
        return canvas

    @master_only
    def add_datasample(
            self,
            name: str,
            image: np.ndarray,
            image_from_to: Sequence[np.array],
            data_sample: Optional[SegDataSample] = None,
            draw_gt: bool = True,
            draw_pred: bool = True,
            show: bool = False,
            wait_time: float = 0,
            # TODO: Supported in mmengine's Viusalizer.
            out_file: Optional[str] = None,
            step: int = 0,
            with_labels: Optional[bool] = False) -> None:
        """Draw datasample and save to all backends.

        - If GT and prediction are plotted at the same time, they are
        displayed in a stitched image where the left image is the
        ground truth and the right image is the prediction.
        - If ``show`` is True, all storage backends are ignored, and
        the images will be displayed in a local window.
        - If ``out_file`` is specified, the drawn image will be
        saved to ``out_file``. it is usually used when the display
        is not available.

        Args:
            name (str): The image identifier.
            image (np.ndarray): The image to draw.
            image_from_to (Sequence[np.array]): The image pairs to draw.
            gt_sample (:obj:`SegDataSample`, optional): GT SegDataSample.
                Defaults to None.
            pred_sample (:obj:`SegDataSample`, optional): Prediction
                SegDataSample. Defaults to None.
            draw_gt (bool): Whether to draw GT SegDataSample. Default to True.
            draw_pred (bool): Whether to draw Prediction SegDataSample.
                Defaults to True.
            show (bool): Whether to display the drawn image. Default to False.
            wait_time (float): The interval of show (s). Defaults to 0.
            out_file (str): Path to output file. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Defaults to True.
        """
        exist_img_from_to = True if len(image_from_to) > 0 else False
        if exist_img_from_to:
            assert len(image_from_to) == 2, '`image_from_to` contains `from` ' \
                'and `to` images'

        classes = self.dataset_meta.get('classes', None)
        palette = self.dataset_meta.get('palette', None)
        semantic_classes = self.dataset_meta.get('semantic_classes', None)
        semantic_palette = self.dataset_meta.get('semantic_palette', None)

        gt_img_data = None
        gt_img_data_from = None
        gt_img_data_to = None
        pred_img_data = None
        pred_img_data_from = None
        pred_img_data_to = None
        
        drawn_img_from = None
        drawn_img_to = None

        if (exist_img_from_to and draw_gt and draw_pred
                and data_sample is not None and 'gt_sem_seg' in data_sample
                and 'pred_sem_seg' in data_sample):
            assert classes is not None, 'class information is not provided ' \
                                        'when visualizing change results.'
            gt_mask = self._draw_mask(data_sample.gt_sem_seg, classes, palette,
                                      with_labels)
            pred_mask = self._draw_mask(data_sample.pred_sem_seg, classes,
                                        palette, with_labels)
            img_from = self._resize_like(image_from_to[0], gt_mask)
            img_to = self._resize_like(image_from_to[1], gt_mask)
            pred_mask = self._resize_like(pred_mask, gt_mask)
            drawn_img = self._make_four_panel(
                (img_from, img_to, gt_mask, pred_mask),
                ('A', 'B', 'GT', 'pred'))

            if show:
                self.show(drawn_img, win_name=name, wait_time=wait_time)
            if out_file is not None:
                mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file)
            else:
                self.add_image(name, drawn_img, None, None, step)
            return

        if draw_gt and data_sample is not None and 'gt_sem_seg' in data_sample:
            gt_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing change ' \
                                        'deteaction results.'
            gt_img_data = self._draw_sem_seg(gt_img_data, data_sample.gt_sem_seg,
                                             classes, palette, with_labels)
        if draw_gt and data_sample is not None and 'gt_sem_seg_from' in data_sample \
            and 'gt_sem_seg_to' in data_sample:
            if exist_img_from_to:
                gt_img_data_from = image_from_to[0]
                gt_img_data_to = image_from_to[1]
            else:
                gt_img_data_from = np.zeros_like(image)
                gt_img_data_to = np.zeros_like(image)
            assert semantic_classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing change ' \
                                        'deteaction results.'
            gt_img_data_from = self._draw_sem_seg(gt_img_data_from,
                                             data_sample.gt_sem_seg_from, semantic_classes,
                                             semantic_palette, with_labels)
            gt_img_data_to = self._draw_sem_seg(gt_img_data_to,
                                             data_sample.gt_sem_seg_to, semantic_classes,
                                             semantic_palette, with_labels)

        if (draw_pred and data_sample is not None
                and 'pred_sem_seg' in data_sample):
            pred_img_data = image
            assert classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing semantic ' \
                                        'segmentation results.'
            pred_img_data = self._draw_sem_seg(pred_img_data,
                                               data_sample.pred_sem_seg,
                                               classes, palette,
                                               with_labels)
            
        if (draw_pred and data_sample is not None and 'pred_sem_seg_from' in data_sample \
            and 'pred_sem_seg_to' in data_sample):
            if exist_img_from_to:
                pred_img_data_from = image_from_to[0]
                pred_img_data_to = image_from_to[1]
            else:
                pred_img_data_from = np.zeros_like(image)
                pred_img_data_to = np.zeros_like(image)
            assert semantic_classes is not None, 'class information is ' \
                                        'not provided when ' \
                                        'visualizing change ' \
                                        'deteaction results.'
            pred_img_data_from = self._draw_sem_seg(pred_img_data_from,
                                             data_sample.pred_sem_seg_from, semantic_classes,
                                             semantic_palette, with_labels)
            pred_img_data_to = self._draw_sem_seg(pred_img_data_to,
                                             data_sample.pred_sem_seg_to, semantic_classes,
                                             semantic_palette, with_labels)

        if (exist_img_from_to and gt_img_data is not None
                and pred_img_data is not None):
            gt_mask = self._draw_mask(data_sample.gt_sem_seg, classes, palette,
                                      with_labels)
            pred_mask = self._draw_mask(data_sample.pred_sem_seg, classes,
                                        palette, with_labels)
            img_from = self._resize_like(image_from_to[0], gt_mask)
            img_to = self._resize_like(image_from_to[1], gt_mask)
            pred_mask = self._resize_like(pred_mask, gt_mask)
            drawn_img = self._make_four_panel(
                (img_from, img_to, gt_mask, pred_mask),
                ('A', 'B', 'GT', 'pred'))
        elif gt_img_data is not None and pred_img_data is not None:
            drawn_img = np.concatenate((gt_img_data, pred_img_data), axis=1)
        elif gt_img_data is not None:
            drawn_img = gt_img_data
        else:
            drawn_img = pred_img_data

        if gt_img_data_from is not None and pred_img_data_from is not None:
            drawn_img_from = np.concatenate((gt_img_data_from, pred_img_data_from), axis=1)
        elif gt_img_data_from is not None:
            drawn_img_from = gt_img_data_from
        else:
            drawn_img_from = pred_img_data_from

        if gt_img_data_to is not None and pred_img_data_to is not None:
            drawn_img_to = np.concatenate((gt_img_data_to, pred_img_data_to), axis=1)
        elif gt_img_data_to is not None:
            drawn_img_to = gt_img_data_to
        else:
            drawn_img_to = pred_img_data_to

        if show:
            if drawn_img_from is not None and drawn_img_to is not None:
                drawn_img_cat = np.concatenate((drawn_img, drawn_img_from, drawn_img_to), axis=0)
                self.show(drawn_img_cat, win_name=name, wait_time=wait_time)
            else:
                self.show(drawn_img, win_name=name, wait_time=wait_time)

        if out_file is not None:
            if drawn_img_from is not None and drawn_img_to is not None:
                drawn_img_cat = np.concatenate((drawn_img, drawn_img_from, drawn_img_to), axis=0)
                mmcv.imwrite(mmcv.bgr2rgb(drawn_img_cat), out_file)
            else:
                mmcv.imwrite(mmcv.bgr2rgb(drawn_img), out_file)
        else:
            self.add_image(name, drawn_img, drawn_img_from, drawn_img_to, step)

    @master_only
    def add_image(self, name: str,
                  image: np.ndarray,
                  image_from: np.ndarray = None,
                  image_to: np.ndarray = None,
                  step: int = 0) -> None:
        """Record the image.

        Args:
            name (str): The image identifier.
            image (np.ndarray, optional): The image to be saved. The format
                should be RGB. Defaults to None.
            step (int): Global step value to record. Defaults to 0.
        """
        for vis_backend in self._vis_backends.values():
            vis_backend.add_image(name, image, image_from, image_to, step)  # type: ignore

    @master_only
    def set_image(self, image: np.ndarray) -> None:
        """Set the image to draw.

        Args:
            image (np.ndarray): The image to draw.
        """
        assert image is not None
        image = image.astype('uint8')
        self._image = image
        self.width, self.height = image.shape[1], image.shape[0]
        # print(image.shape)
        self._default_font_size = max(
            np.sqrt(self.height * self.width) // 90, 10)

        self.fig_save.set_size_inches(  # type: ignore
            self.width / self.dpi, self.height / self.dpi)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        self.ax_save.cla()
        self.ax_save.axis(False)
        self.ax_save.imshow(
            image,
            extent=(0, self.width, self.height, 0),
            interpolation='none')
