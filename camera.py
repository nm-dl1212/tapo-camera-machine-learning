import rtsp
from onvif import ONVIFCamera
import time
import cv2  
import threading

class CamStream(object):
    def __init__(self, **kwargs):
        self.user = kwargs["user"]
        self.pwd = kwargs["pwd"]
        self.ipaddr = kwargs["ipaddr"]
        self.port = kwargs["port"]
        self.stream = kwargs["stream"]
        self.onvif_port = kwargs["onvif_port"]
        self.is_connect = False

        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.cap = None

    def open(self):
        rtsp_url = f"rtsp://{self.user}:{self.pwd}@{self.ipaddr}:{self.port}/{self.stream}"
        self.cap = cv2.VideoCapture(rtsp_url)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream")
        self.running = True
        t = threading.Thread(target=self._capture_thread, daemon=True)
        t.start()

    def _capture_thread(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            else:
                time.sleep(0.05)

    def read(self):
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            else:
                return None

    def close(self):
        self.running = False
        if self.cap:
            self.cap.release()


class CamPtz(CamStream):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
        self.XMAX = 1
        self.XMIN = -1
        self.YMAX = 1
        self.YMIN = -1
        self.moverequest = None
        self.ptz = None
        self.is_ptz_active = False


    def setup_ptz(self):
        mycam = ONVIFCamera(self.ipaddr, self.onvif_port, self.user, self.pwd)
        # Create media service object
        media = mycam.create_media_service()
        # Create ptz service object
        self.ptz = mycam.create_ptz_service()
        # Get target profile
        media_profile = media.GetProfiles()[0]
        # Get PTZ configuration options for getting continuous move range
        request = self.ptz.create_type('GetConfigurationOptions')
        request.ConfigurationToken = media_profile.PTZConfiguration.token
        ptz_configuration_options = self.ptz.GetConfigurationOptions(request)

        self.moverequest = self.ptz.create_type('ContinuousMove')
        self.moverequest.ProfileToken = media_profile.token
        if self.moverequest.Velocity is None:
            self.moverequest.Velocity = self.ptz.GetStatus({'ProfileToken': media_profile.token}).Position

        # Get range of pan and tilt
        # NOTE: X and Y are velocity vector
        self.XMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Max
        self.XMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].XRange.Min
        self.YMAX = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Max
        self.YMIN = ptz_configuration_options.Spaces.ContinuousPanTiltVelocitySpace[0].YRange.Min
        return


    def move(self, x, y):
        self.moverequest.Velocity.PanTilt.x = x
        self.moverequest.Velocity.PanTilt.y = y
        if self.is_ptz_active:
            self.ptz.Stop({'ProfileToken': self.moverequest.ProfileToken})
        active = True
        self.ptz.ContinuousMove(self.moverequest)