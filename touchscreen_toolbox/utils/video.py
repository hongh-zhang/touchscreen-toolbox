import cv2


def map_video(
        func,
        video_in: str,
        video_out: str,
        fourcc: str = "mp4v",
        fps: int = 25,
        dim=(640, 480),
):
    """
    Map video with the given [func]

    Args
    --------
    video_in : str
        path to input video

    video_out : str
        path to output video

    func : (:: ndarray -> ndarray)
        function to be mapped to each frame

    """

    try:
        # initialize opencv
        cap = cv2.VideoCapture(video_in)
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(video_out, fourcc, fps, dim)

        # iterate each frame and apply function
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame = func(frame, count)
                writer.write(frame)
            else:
                break
            count += 1

    except Exception as exc:
        raise exc

    # release file before terminating
    finally:
        cap.release()
        writer.release()


def text_writer(data, col, position=(0, 0), fontScale=1):
    """
    Blahblahblah
    
    * data should be indexed by frame number
    """

    data = data[col]
    frame_start_end = (data.index[0], data.index[-1])

    def func(frame, count):
        if count < frame_start_end[0] or count > frame_start_end[-1]:
            return frame

        # read text from dataframe
        text = f"{col}: {data.loc[count]}"

        # calculate text size and set position
        text_size = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=fontScale, thickness=1)
        height = text_size[0][1]
        position2 = (position[0], int(position[1] + height * 1.5))

        # write
        frame = cv2.putText(frame, text, position2, cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 1)
        return frame

    return func
