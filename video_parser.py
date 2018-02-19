import cv2
import numpy as np
import argparse
from label_parser import label_reader

class video_reader(object):
    can="/can_video.avi"
    depth="/depth_video.avi"
    can_reshape=(300, 48)
    depth_reshape=(640, 60)
    def __init__(self,args):

        self.video_file=args.video_file
        self.out_dir=args.output_dir
        self.skip_frame=args.start_frame
        self.duration=args.duration
        self.label=label_reader(args)
        self.isLabeling=args.labeling
        self.isIndividual=args.saving_individual
        self.isViz=args.visualization

        self.getNames()
        self.config_reader_writer()
        self.skip()


    def getNames(self):
        chunks = self.video_file.split('/')
        dirname="/".join(chunks[:-1])
        self.video="/"+chunks[len(chunks)-1]
        self.can_file=dirname+self.can
        self.depth_file=dirname+self.depth



    def config_reader_writer(self):

        # reader configuration
        full_names = [self.video_file, self.can_file, self.depth_file]
        base_names = [self.video, self.can, self.depth]
        self.caps = [cv2.VideoCapture(nam) for nam in full_names]

        # writer configuration
        self.writers=[]
        for i,cap in enumerate(self.caps):
            if (cap.isOpened() == False):
                print("Error opening video stream or file")
                return -1
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            if (self.isIndividual):
                self.writers.append(cv2.VideoWriter(self.out_dir+base_names[i], cv2.cv.CV_FOURCC(*'XVID'), 10, (frame_width, frame_height)))
            if (i==0):
                self.combine_writer=cv2.VideoWriter(self.out_dir+"/combine_video.avi", cv2.cv.CV_FOURCC(*'XVID'), 10, (frame_width, frame_height))

    def skip(self):
        for cap in self.caps:
            cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, self.skip_frame)

    def run(self):
        for _ in range(self.duration):
            main_image=self.transform()
            main_image = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)
            if self.isLabeling:
                data, label = next(self.label)
                cv2.polylines(main_image, [data], False, (0, 0, 255), 2)
                cv2.polylines(main_image, [label], False, (180, 105, 255), 2)

            cv2.imshow("resized", main_image)
            cv2.waitKey(100)


    def next(self):
        images=[]
        for i,cap in enumerate(self.caps):
            ret, frame = cap.read()
            if (self.isIndividual):
                self.writers[i].write(frame)
            images.append(frame)
        return images
    def transform(self):
        main_image,can_image,depth_imge = next(self)


        # can_resized = cv2.resize(can_image, self.can_reshape)
        can_image = cv2.cvtColor(can_image, cv2.COLOR_BGR2GRAY)
        xdata = [y * np.ones((3, 3)) for x in can_image for y in x]
        can_resized = np.array(xdata).reshape(self.can_reshape)
        can_resized = cv2.cvtColor(can_resized, cv2.COLOR_GRAY2BGR)

        depth_resized = cv2.resize(depth_imge, self.depth_reshape)




        h, w, c = np.shape(main_image)
        #fix the transformation manually
        main_image[h - 128:h - 80, 170:470, :] = can_resized
        main_image[h - 60:h, :, :] = depth_resized
        main_image=cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)

        return main_image

    def save(self):
        if(self.isIndividual):
            print('video files are saving individually')
        else:
            print('saving combine videos only')


        for i,cap in enumerate(self.caps):
            cap.release()
            if (self.isIndividual):
                self.writers[i].release()
        self.combine_writer.release()
        cv2.destroyAllWindows()









if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Labeling dataset')

    args.add_argument('--video_file',type=str, default='/home/redwan/Server/depth_can/dataset/ELECOM_V2/ELECOM_20160319_134804_144304/camera_video.avi')
    # args.add_argument('--video_file', type=str,
    #                   default='/home/redwan/Server/depth_can/dataset/ELECOM_V2/ELECOM_20160319_134804_144304/camera_video.avi')

    args.add_argument('--output_dir',type=str, default='/home/redwan/Server/depth_can/results/test/ELECOM_20160319_134804_144304/data/')
    args.add_argument('--lable_file',type=str, default='/home/redwan/Server/depth_can/results/test/ELECOM_20160319_134804_144304/LSTM_GN.csv')
    args.add_argument('--start_frame', type=int, default=20400)
    args.add_argument('--duration',type=int, default=1000)
    args.add_argument('--visualization', type=bool, default=True)
    args.add_argument('--labeling', type=bool, default=True)
    param = args.parse_args()

    cr=video_reader(param)
    cr.run()







