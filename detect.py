#!==============================================================================

#! Please run these commands before testing the script
#! pip uninstall opencv-python-headless opencv-contrib-python opencv-python opencv-contrib-python-headless
#! pip install opencv-python-headless
#! sudo apt get install libgtk2.0-dev pkg-config

#!==============================================================================
from threading import Thread
import argparse
from utils.plate_rotation_adjust import *
from utils.datasets import *
from utils.utils import *
from pathlib import Path
from utils.wrap_prespective import make_plate_upright
from utils.centroid_tracking import CentroidTracker
from predict import init_model_and_vocab, predict_characters, OUTPUT_TXT_FILE
import mimetypes
import torch

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from datetime import datetime

from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QDateTime, Qt, QTimer, QObject
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
                             QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QProgressBar, QPushButton, QRadioButton, QScrollBar, QSizePolicy,
                             QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit, QVBoxLayout, QDesktopWidget)

# Imports for enhancing cropped images
from iofile import DataLoader
from RRDBNet import RRDBNet
from glob import glob
import numpy as np
import cv2
import os
import tensorflow as tf

mimetypes.init()


lock = QtCore.QMutex()
upperImage = None
bottomImage = None
mainImage = None


DETECTED_PLATE_PARTS = []
TEXT_PER_PLATE = []
MAX_NUMBER_OF_FRAMES_PLATE_CAN_BE_INVISIBLE = 4
MIN_PLATE_AREA = 500


def save_roi(model1, tl, br, original_image, save_path, check_roi, draw_center_line=True, ):

    # tl ==> (x_value, y_value)

    #! crop the roi region from the original image
    roi = original_image[tl[1]:br[1], tl[0]:br[0]]

    # ! it needs more improvements that I'll do later
    roi = make_plate_upright(roi)
    height, width = roi.shape[:2]

    #! check if the ROI angel
    """
    the rectangle is the roi image after applying wrap perspective
    "@" is the angel value. 
    from other plates I see that this angel value is between 25 : 40 degrees
    ----------
    |    \   |
    |    @ \ |
    ----------
    """
    angel = math.degrees(math.atan(height/width))
    if check_roi and (angel < 25 or angel > 40):
        return False

    y = height//2 - 6
    copy = roi.copy()
    copy = enhancerSRGAN(copy, model1)
    upper_half, lower_half = copy[:y, :], copy[y:, :]

    #! if not all the image dimensions are bigger than zero
    #! then don't add them to the images list and don't save the original plate image
    for x in [upper_half, lower_half]:
        if not all(x.shape):
            return

    DETECTED_PLATE_PARTS.extend([upper_half, lower_half])

    #! draw a central horizontal line
    if draw_center_line:
        cv2.line(roi, (0, y), (width, y), (0, 255, 0), 1)

    cv2.imwrite(save_path, roi)  # ! save the image
    print(f"\n[INFO] SAVING IMAGE: {save_path}")
    return save_path

def load_weights_into_RRDB():
    MODEL_PATH = r'X:/Upwork/projects/ir_dnr/weights/rrdb'
    model1 = RRDBNet(blockNum=10)
    model1.built = True
    model1.load_weights(MODEL_PATH)

    return model1
  
# Please check folder paths before executing detect.py
def enhancerSRGAN(ori, model1):
    #RESULTS_PATH = r'X:/Upwork/projects/ir_dnr/results'
    # pretrained rrdb network weights
    row, col, dep = ori.shape
    ori = ori[np.newaxis,: , :, :]
    ori = tf.image.convert_image_dtype(ori, tf.float32)

    yPred = model1.predict(ori)
    img = yPred[0]
    predi = np.dstack((img[:,:,0], img[:,:,1], img[:,:,2]))
    predi = np.clip((predi*255), 0, 255)
    predi = predi.astype(np.uint8)
    predi = cv2.resize(predi, dsize=(col, row), interpolation=cv2.INTER_CUBIC)
    
    #cv2.imwrite(os.path.join(RESULTS_PATH, str(counter), ".png"), predi)
    #counter += 1
    return predi


def setMainImage(img):
    lock.lock()
    global mainImage
    mainImage = img.copy()
    lock.unlock()


def setUpperImage(img):
    lock.lock()
    global upperImage
    upperImage = img.copy()
    lock.unlock()


def setBottomImage(img):
    lock.lock()
    global bottomImage
    bottomImage = img.copy()
    lock.unlock()


def getMainImage():
    lock.lock()
    global mainImage
    mainImage1 = mainImage.copy()
    lock.unlock()
    return mainImage1


def getUpperImage():
    lock.lock()
    global upperImage
    upperImage1 = upperImage.copy()
    lock.unlock()
    return upperImage1


def getBottomImage():
    lock.lock()
    global bottomImage
    bottomImage1 = bottomImage.copy()
    lock.unlock()
    return bottomImage1


thread_start = False
stop_flag = False


def getStop():
    lock.lock()
    global stop_flag
    b = stop_flag
    lock.unlock()
    return b


def setStop(b):
    lock.lock()
    global stop_flag
    stop_flag = b
    lock.unlock()


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def toQImage(im, copy=False):
    #! sometime the image is delivered corrupt with the firest axis value below 5
    #! so we check this and refuse it
    if im is None or im.shape[1] < 10:
        return QImage()
    try:

        if im.dtype == np.uint8:
            if len(im.shape) == 2:
                qim = QImage(
                    im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_Indexed8)
                qim.setColorTable(gray_color_table)
                return qim.copy() if copy else qim

            elif len(im.shape) == 3:
                if im.shape[2] == 3:
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    qim = QImage(
                        im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_RGB888)
                    return qim.copy() if copy else qim
                elif im.shape[2] == 4:
                    qim = QImage(
                        im.data, im.shape[1], im.shape[0], im.strides[0], QImage.Format_ARGB32)
                    return qim.copy() if copy else qim

    except Exception as e:
        print(f"[Error]: {e}")
        return QImage()


def detect(monitor, save_img=True):
    global DETECTED_PLATE_PARTS, TEXT_PER_PLATE, MAX_NUMBER_OF_FRAMES_PLATE_CAN_BE_INVISIBLE, MIN_PLATE_AREA

    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    prediction_phase_model, vocab = init_model_and_vocab()

    #! Remove all the files in the output folder
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    #! empty the file before prediction
    with open(OUTPUT_TXT_FILE, "w", encoding="UTF-8") as txt:
        pass
    # Loading weights into image enhancing model  
    model1 = load_weights_into_RRDB()

    total_number_fo_plates = 0  # !  total number of recognized plates
    tracker = CentroidTracker(
        maxDisappeared=MAX_NUMBER_OF_FRAMES_PLATE_CAN_BE_INVISIBLE,
        minArea=MIN_PLATE_AREA)

    # Initialize
    device = torch_utils.select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    plate_history = []
    my_thread = Thread(target=predict_characters, args=(
        prediction_phase_model,
        vocab,
        DETECTED_PLATE_PARTS,
        TEXT_PER_PLATE
    ))

    # Load model
    google_utils.attempt_download(weights)
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32

    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(
            name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load(
            'weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    fileKind = ""

    if webcam:
        view_img = True
        # set True to speed up constant image size inference
        torch.backends.cudnn.benchmark = True

        if RepresentsInt(source):
            fileKind = "webcam"
        else:
            fileKind = "ipcamera"
        dataset = LoadStreams(source, img_size=imgsz)

    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.names if hasattr(model, 'names') else model.modules.names
    colors = [[random.randint(0, 255) for _ in range(3)]
              for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    frameNum = 0
    prevPath = ""

    for path, img, im0s, vid_cap in dataset:
        b = getStop()

        if b == True:
            break
        if prevPath != path:
            prevPath = path
            frameNum = 0
        if fileKind != "webcam" and fileKind != "ipcamera":
            try:
                mimestart = mimetypes.guess_type(path)[0]
                if mimestart != None:
                    mimestart = mimestart.split('/')[0]
                    # if mimestart == 'audio' or mimestart == 'video' or mimestart == 'image':
                    if mimestart == 'video':
                        fileKind = mimestart
                    elif mimestart == 'image':
                        fileKind = mimestart
            except:
                fileKind = ""
        if fileKind == "":
            continue
        frameNum += 1
        if fileKind == "webcam" or fileKind == "ipcamera" or fileKind == "video":
            view_img = True
        else:
            view_img = True

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()  # Waits for everything to finish running
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   fast=True, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = torch_utils.time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            b = getStop()
            if b == True:
                # print ('break!!!!!!!!!!!!')
                break

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string

            # Â normalization gain whwh
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            plates_was_seen_before = []

            # ! this is important so that we can delete plates that disappeared
            if det is None or len(det) == 0:
                plates_was_seen_before = tracker.track([])

            else:
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                #! ------------------------ Tracking Plates ------------------------
                #! check if the angel of the roi is valid or not
                #! don't check if the roi is valid if the input data was images
                check_roi = False

                #! IT"S SO IMPROTATNT to call the block of code below right here
                #! because we resize the input image to small dimensions for the model
                #! then we need to rescale the input back to it's original size
                #! and the line above does this task and also
                #! this rescaling is VERY IMPORTANT because we want the check the area of the plate
                #! on the original image
                if fileKind != 'image':  # ! don't apply on images
                    start = time.time()
                    check_roi = True
                    plates_was_seen_before = tracker.track(det)
                    print(f'\n[INFO]: Tracker Toke: {time.time() - start}s')

                #! ------------------------ Processing Plates ------------------------

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                tl, br = None, None  # ! vertices of bounding rectaangel for the licence plate
                # Write results
                for plate_number, current_detection in enumerate(det):
                    #! the star in the begining of the variable "xyxy" means that
                    #! this variable will take all the values in "current_detection"
                    #! except for the last two ones
                    *xyxy, conf, cls = current_detection

                    b = getStop()
                    if b == True:
                        # print ('break!!!!!!!!!!!!')
                        break

                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)
                                          ) / gn).view(-1).tolist()  # normalized xywh
                        with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') %
                                       (cls, *xywh))  # label format

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)
                        modified_img, tl, br = plot_one_box(xyxy, im0, label=label,
                                                            color=colors[int(cls)], line_thickness=3)

                        #! Save the ROI Image
                        #!=====================
                        #! ROI Save path

                        if check_roi and len(plates_was_seen_before) > 0 and plates_was_seen_before[plate_number]:
                            print("\nSKIPPING...\n")
                            # continue
                        # if len(plates_was_seen_before) > 0 and all(plates_was_seen_before):
                        #     print("\nSKIPPING...\n")
                        #     continue
                        else:
                            total_number_fo_plates += 1

                            path_without_ext, img_ext = os.path.splitext(
                                save_path)[0], ".png"

                            roi_save_path = path_without_ext + img_ext

                            roi_save_path = f"{os.sep}#{total_number_fo_plates}_ROI_{time.time()}".join(
                                roi_save_path.rsplit(os.sep, 1))

                            roi_save_path = save_roi(
                                model1, tl, br, im0, roi_save_path, check_roi, draw_center_line=False)

                            print("\nNEW IMAGE...\n")
                            #! this plate is not valid so don't count it we subtract here
                            #! because at the begining we increased the count by 1
                            if not roi_save_path:
                                total_number_fo_plates += -1

                            if not my_thread.is_alive():
                                print("\n[INFO]: STARTING THE THREAD")
                                my_thread = Thread(target=predict_characters, args=(
                                    prediction_phase_model,
                                    vocab,
                                    DETECTED_PLATE_PARTS,
                                    TEXT_PER_PLATE
                                ))

                                my_thread.start()

                        #! make the original input image equal to the image that we drew and put information on
                        im0 = modified_img

            # Print time (inference + NMS)
            time_taken = t2 - t1
            fps = int(1 / time_taken)
            print('%sDone. (%.3fs)  FPS: %d' % (s, time_taken, fps))

            enhancerSRGAN()  # Image enhancing function

            #! ----------------------- SHOW RESULTS ON GUI -----------------------
            #! VERY IMPORTANT: we should update the GUI with every frame

            #! DETECTED_PLATE_PARTS = [upper_#1, lower_#1, upper_#2, lower_#2, ...]
            if len(DETECTED_PLATE_PARTS) > 1:
                upperMat = DETECTED_PLATE_PARTS[-2].copy()
                bottomMat = DETECTED_PLATE_PARTS[-1].copy()
                setUpperImage(upperMat)
                list_1 = ['upperimage', '']
                monitor.detect_signal.emit(list_1)

                setBottomImage(bottomMat)
                list_2 = ['bottomimage', '']
                monitor.detect_signal.emit(list_2)

            # ! TEXT_PER_PLATE = [['豊橋55', 'せ1954'], ["upper", "lower"], ...]
            if len(TEXT_PER_PLATE) > 0:
                upperstring, bottomstring = TEXT_PER_PLATE[-1]
                list_3 = ['upperstr', upperstring]
                monitor.detect_signal.emit(list_3)

                list_4 = ['bottomstr', bottomstring]
                monitor.detect_signal.emit(list_4)

            # Stream results
            if view_img:
                setMainImage(im0)
                main_list = ['mainimage', '']
                monitor.detect_signal.emit(main_list)

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    input_ext = os.path.splitext(save_path)[-1].lower()
                    list_ext = [".png", ".jpg", ".jpeg"]
                    if not input_ext:
                        save_path += ".png"
                    #! this a fix for the Network Camera invalid extension
                    elif not (input_ext.lower() in list_ext):
                        save_path = os.path.splitext(save_path)[0] + ".png"
                    cv2.imwrite(save_path, im0)

                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    while(my_thread.is_alive()):
        time.sleep(0.001)

    print('[THREADING]: Is Done.\n')
    print("Plates Text", "="*30, sep="\n")
    for i, t in enumerate(TEXT_PER_PLATE):
        upper, lower = t
        print(f"\n#{i+1}\t\tUpper: {upper}\t\tLower: {lower}\n")

    RESULTS_PATH = " " 
    if os.path.exists(RESULTS_PATH):
        shutil.rmtree(RESULTS_PATH) 
    os.makedirs(RESULTS_PATH)     


class UIMonitor(QObject):

    detect_signal = QtCore.pyqtSignal(list)

    @QtCore.pyqtSlot()
    def monitor_images(self):
        list = ['started', '']
        self.detect_signal.emit(list)
        global thread_start
        thread_start = True
        detect(self)
        thread_start = False
        list = ['stopped', '']
        self.detect_signal.emit(list)


class WidgetGallery(QDialog):

    @QtCore.pyqtSlot(list)
    def image_callback(self, list):
        type = list[0]
        if type == 'stopped':
            self.startScriptButton.setText("Start D&R License Plate")
            self.thread.quit()
        elif type == 'started':
            self.startScriptButton.setText("Stop D&R License Plate")

        elif type == 'upperimage':
            mat = getUpperImage()
            uppperImagePlate = toQImage(mat)
            self.drawUpperLabel.setScaledContents(True)
            self.drawUpperLabel.setPixmap(QtGui.QPixmap(uppperImagePlate))
        elif type == 'bottomimage':
            mat = getBottomImage()
            bottomImagePlate = toQImage(mat)
            self.drawBottomLabel.setScaledContents(True)
            self.drawBottomLabel.setPixmap(QtGui.QPixmap(bottomImagePlate))
        elif type == 'mainimage':
            mat = getMainImage()
            image = toQImage(mat)
            self.drawLabel.setScaledContents(True)
            self.drawLabel.setPixmap(QtGui.QPixmap(image))
        elif type == 'upperstr':
            # ! our list will look like that: ['upperstr', '三河5']
            str = list[1]

            self.labelCharactersRecognitionUpper.setText(str)
        elif type == 'bottomstr':
            str = list[1]
            self.labelCharactersRecognitionBottom.setText(str)

    def on_start_clicked(self):

        global thread_start
        if thread_start == False:
            setStop(False)
            self.ui_monitor.moveToThread(self.thread)
            self.thread.start()
        else:
            setStop(True)
            print('stopped Clicked!')

    def closeEvent(self, event):
        print("X is clicked")
        self.bStart = False
        #! this is very important because cv2.waitKey causes errors with pyqt5
        raise StopIteration
        sys.exit(app.exec_())

    def __init__(self, parent=None):
        super(WidgetGallery, self).__init__(parent)

        self.ui_monitor = UIMonitor()
        self.thread = QtCore.QThread(self)
        self.thread.started.connect(self.ui_monitor.monitor_images)
        self.ui_monitor.detect_signal.connect(self.image_callback)

        self.originalPalette = QApplication.palette()

        self.resize(850, 600)
        self.center()

        self.drawLabel = QLabel(
            'Real-time streaming for License Plate Detection')
        self.drawUpperLabel = QLabel('Upper')
        self.drawBottomLabel = QLabel('Bottom')
        self.labeldate = QLabel('')
        self.labeltime = QLabel('')
        self.labelAllDate = QLabel('')
        self.labelCharactersRecognitionUpper = QLabel('')
        self.labelCharactersRecognitionBottom = QLabel('')

        self.createTopLayOut()
        self.createBottomLeftLayOut()
        self.createBottomRightLayOut()

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(self.topLayOut, 2)

        bottomLayOut = QHBoxLayout()
        bottomLayOut.addLayout(self.bottomLeftLayOut, 4)
        bottomLayOut.addLayout(self.bottomRightLayOut, 1)
        mainLayout.addLayout(bottomLayOut, 5)

        bottomLayOut1 = QHBoxLayout()
        mainLayout.addLayout(bottomLayOut1, 1)

        self.setLayout(mainLayout)
        self.start_timer()
        self.setWindowTitle("Japan License Plate Detection and Recognition")
        self.bStart = False

    def createTopLayOut(self):
        self.topLayOut = QVBoxLayout()

        self.startScriptButton = QPushButton("Start D&R License Plate")
        self.startScriptButton.setDefault(True)
        self.startScriptButton.clicked.connect(self.on_start_clicked)

        self.startScriptButton.setSizePolicy(
            QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.startScriptButton.setFixedWidth(450)
        self.startScriptButton.setFixedHeight(60)
        self.startScriptButton.setFont(QFont('Times', 15))

        line = QLabel()
        line.setFixedWidth(820)
        line.setFixedHeight(2)
        line.setStyleSheet("border: 1px solid gray;")
        line.setStyleSheet("background-color: gray")

        interLabel1 = QLabel('')
        interLabel2 = QLabel('')

        self.topLayOut .addWidget(interLabel1, 1)
        self.topLayOut .addWidget(self.startScriptButton, 3)
        self.topLayOut .addWidget(interLabel2, 1)
        self.topLayOut .addWidget(line)

    def createBottomLeftLayOut(self):
        self.bottomLeftLayOut = QVBoxLayout()
        interLabel1 = QLabel('')
        interLabel2 = QLabel('')

        self.drawLabel.setFixedWidth(600)
        self.drawLabel.setFixedHeight(400)
        self.drawLabel.setStyleSheet("border: 1px solid black;")
        self.drawLabel.setStyleSheet("background-color: black")

        self.bottomLeftLayOut.addWidget(interLabel1, 1)
        self.bottomLeftLayOut.addWidget(self.drawLabel)
        self.bottomLeftLayOut.addWidget(interLabel2, 1)

    def createBottomRightLayOut(self):

        self.drawUpperLabel.setFixedWidth(200)
        self.drawUpperLabel.setFixedHeight(80)

        self.drawUpperLabel.setStyleSheet("border: 1px solid black;")
        self.drawUpperLabel.setStyleSheet("background-color: black")

        self.drawBottomLabel.setFixedWidth(200)
        self.drawBottomLabel.setFixedHeight(80)
        self.drawBottomLabel.setStyleSheet("border: 1px solid black;")
        self.drawBottomLabel.setStyleSheet("background-color: black")

        labeltop1 = QLabel('')
        labeltop2 = QLabel('')
        labeltop3 = QLabel('')
        labeltop4 = QLabel('')
        labeltop5 = QLabel('')
        labelbottom = QLabel('')

        labelRecognition = QLabel('Characters Recognition')
        labelRecognition.setFont(QFont('Times', 15))

        labelCurrentTime = QLabel('CurrentTimeWhat')
        labelCurrentDate = QLabel('Date')

        self.labelCharactersRecognitionUpper.setFixedHeight(40)
        self.labelCharactersRecognitionUpper.setFont(QFont('Times', 15))
        self.labelCharactersRecognitionUpper.setStyleSheet(
            "border: 1px solid blue;")

        self.labelCharactersRecognitionBottom.setFixedHeight(40)
        self.labelCharactersRecognitionBottom.setFont(QFont('Times', 15))
        self.labelCharactersRecognitionBottom.setStyleSheet(
            "border: 1px solid blue;")

        labelCurrentTime.setFixedHeight(30)
        labelCurrentTime.setFont(QFont('Times', 15))
        self.labeltime.setFixedHeight(40)
        self.labeltime.setFont(QFont('Times', 15))
        self.labeltime.setStyleSheet("border: 1px solid black;")

        labelCurrentDate.setFixedHeight(30)
        labelCurrentDate.setFont(QFont('Times', 15))
        self.labeldate.setFixedHeight(40)
        self.labeldate.setFont(QFont('Times', 15))
        self.labeldate.setStyleSheet("border: 1px solid black;")

        self.bottomRightLayOut = QVBoxLayout()
        self.bottomRightLayOut.addWidget(labeltop1, 5)
        self.bottomRightLayOut.addWidget(self.drawUpperLabel)
        self.bottomRightLayOut.addWidget(self.drawBottomLabel)
        self.bottomRightLayOut.addWidget(labeltop4, 1)
        self.bottomRightLayOut.addWidget(labelRecognition, 2)
        self.bottomRightLayOut.addWidget(self.labelCharactersRecognitionUpper)
        self.bottomRightLayOut.addWidget(labeltop5, 1)
        self.bottomRightLayOut.addWidget(self.labelCharactersRecognitionBottom)

        self.bottomRightLayOut.addWidget(labeltop3, 2)
        self.bottomRightLayOut.addWidget(labelCurrentTime, 1)
        self.bottomRightLayOut.addWidget(self.labeltime)

        self.bottomRightLayOut.addWidget(labelCurrentDate)
        self.bottomRightLayOut.addWidget(self.labeldate)

        self.bottomRightLayOut.addWidget(labelbottom, 1)

    def start_timer(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.setInterval(1000)
        self.timer.start()

    def update(self):
        nowdate = datetime.now().strftime('%d/%m/%Y')
        nowtime = datetime.now().strftime('%H:%M:%S')
        self.labeldate.setText(nowdate)
        self.labeltime.setText(nowtime)

    def center(self):
        # geometry of the main window
        qr = self.frameGeometry()

        # center point of screen
        cp = QDesktopWidget().availableGeometry().center()

        # move rectangle's center point to screen's center point
        qr.moveCenter(cp)

        # top left of rectangle becomes top left of window centering it
        self.move(qr.topLeft())


if __name__ == '__main__':

    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='pretrained/last.pt', help='model.pt path')
    # file/folder, 0 for webcam sample_cars/, rtsp://username:password@ip_address/axis-media/media.amp
    parser.add_argument(
        # '--source', type=str, default='rtsp://username:password@IP_address/axis-media/media.amp', help='source')
        '--source', type=str, default='0', help='source')
    # '--source', type=str, default="/inference/images", help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False,
                        action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+',
                        type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')

    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    app = QApplication(sys.argv)
    gallery = WidgetGallery()
    gallery.show()
    sys.exit(app.exec_())
