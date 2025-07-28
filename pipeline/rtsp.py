import av

import pipeline as pl


class RTSPSource(pl.Component):
    def __init__(self, url: str):
        super().__init__()
        self.url = url

    def pipeline_thread_init(self):
        '''
        Initialize the RTSP input stream.
        '''
        self.container = av.open(self.url)
        self.stream = self.container.decode(video=0)

    def process(self, data):
        '''
        Reads a frame from the RTSP stream.
        '''
        data.frame = next(self.stream).to_image()  # Convert to PIL Image


class RTSPSink(pl.Component):
    def __init__(self, url: str, codec: str = 'h264'):
        super().__init__()
        self.url = url
        self.codec = codec

    def pipeline_thread_init(self):
        '''
        Initialize the RTSP output stream.
        '''
        self.output = av.open(self.url, mode='w', format='rtsp')
        self.output.add_stream(self.codec)

    def process(self, data):
        '''
        Writes data.frame to an RTSP server.
        '''
        frame = data.frame
        packet = self.output.encode(frame)
        if packet:
            self.output.mux(packet)
