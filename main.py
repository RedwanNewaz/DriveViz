import numpy as np
import cv2
from video_parser import video_reader
from multiprocessing.pool import ThreadPool
from collections import deque
import argparse


class Joint_View(object):
    can_reshape=(300,80)
    depth_reshape=(640, 60)

    def resize_can(self,can_image):
        scale_factor=(3,5)#INDEX REVERSED (100X3,16X5)
        can_image = cv2.cvtColor(can_image, cv2.COLOR_BGR2GRAY)
        xdata = [y * np.ones(scale_factor) for x in can_image for y in x]
        can_resized = np.array(xdata,dtype=np.uint8).reshape(self.can_reshape)
        can_resized = cv2.cvtColor(can_resized.T, cv2.COLOR_GRAY2BGR)
        return can_resized

    def transform(self, frames):
        main_image, can_image, depth_imge = frames
        depth_resized = cv2.resize(depth_imge, self.depth_reshape)
        h, w, c = np.shape(main_image)
        # fix the transformation manually
        main_image[h - 128:h - 48, 170:470, :] = self.resize_can(can_image)
        main_image[h - 60:h, :, :] = depth_resized
        # main_image = cv2.cvtColor(main_image, cv2.COLOR_BGR2GRAY)
        # main_image = cv2.cvtColor(main_image, cv2.COLOR_GRAY2BGR)

        return main_image



class combine_frames(video_reader):




    def run(self):
        threadn = cv2.getNumberOfCPUs()-1
        pool = ThreadPool(processes=threadn)
        process_frame= Joint_View()
        pending = deque()


        cv2.namedWindow('Threaded Video', cv2.WINDOW_NORMAL)
        for _ in range(self.duration):

            frame=next(self)
            task = pool.apply_async(process_frame.transform, (frame,))
            pending.append(task)
        print("processing done")
        for _ in range(self.duration):

            if len(pending) > 0 and pending[0].ready():
                result = pending.popleft().get()
                if self.isLabeling:
                    data, label = next(self.label)
                    cv2.polylines(result, [data], False, (0, 0, 255), 2)
                    cv2.polylines(result, [label], False, (180, 105, 255), 2)


                self.combine_writer.write(result)
                if(self.isViz):
                    cv2.imshow('Threaded Video', result)
                    cv2.waitKey(100)




if __name__ == '__main__':

    args = argparse.ArgumentParser(description='Labeling dataset')
    name='ELECOM_20160319_123014_133014'# folder name

    args.add_argument('--video_file',type=str, default='/home/redwan/Server/depth_can/dataset/ELECOM_V2/%s/camera_video.avi'%name)
    args.add_argument('--output_dir',type=str, default='/home/redwan/Server/depth_can/results/test/%s/ex2'%name)
    args.add_argument('--lable_file',type=str, default='/home/redwan/Server/depth_can/results/test/%s/ex2/CNN_GN.csv'%name)
    args.add_argument('--start_frame', type=int, default=8200) #33300
    args.add_argument('--duration',type=int, default=1000)
    args.add_argument('--visualization', type=bool, default=True)
    args.add_argument('--labeling', type=bool, default=False)
    args.add_argument('--saving_individual', type=bool, default=False)

    param = args.parse_args()
    cr=combine_frames(param)
    cr.run()
    cr.save()
