import cv2
import os


class FpsChanger:
    def __init__(self, data_folder: str, fps_folder: str, target_fps: int):
        self.DATA_DIR = data_folder
        self.FPS_DIR = fps_folder
        self.fps = target_fps
        os.makedirs(self.FPS_DIR, exist_ok=True)

        while True:
            set_list = [set(os.listdir(self.DATA_DIR)), set(os.listdir(self.FPS_DIR))]
            #[s.remove(".DS_Store") for s in set_list if ".DS_Store"]
            self.CHECKER = set_list[0] - set_list[1]

            if not self.CHECKER:
                break

            video_paths = [
                os.path.join(self.DATA_DIR, video_file)
                for video_file in list(self.CHECKER)
            ]
            self.change(video_paths=video_paths)

    def change(self, video_paths: list):
        for video_path in video_paths:
            fps_path = os.path.join(self.FPS_DIR, os.path.split(video_path)[1])

            cap = cv2.VideoCapture(video_path)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out = cv2.VideoWriter(fps_path, fourcc, self.fps, (width, height))

            while cv2.waitKey(1):
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(
                    cv2.CAP_PROP_FRAME_COUNT
                ):
                    print(f"{video_path} -> {fps_path}")
                    break

                ret, frame = cap.read()
                out.write(frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    fps_changer = FpsChanger(
        data_folder="/home/ubuntu/nia/Final_Test/data/Train/Train/Video",
        fps_folder="/home/ubuntu/nia/Final_Test/data/Train/Train/Video_fps",
        target_fps=25,
    )
    del fps_changer
