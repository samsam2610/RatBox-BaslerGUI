import wx


class CameraControl(wx.Panel):
    """
    Reusable camera control panel.
    Each instance could represent a different physical camera.
    """
    def __init__(self, parent, camera_name="Camera"):
        super().__init__(parent)

        # Vertical layout for this camera
        vbox = wx.BoxSizer(wx.VERTICAL)

        # Title / label
        title = wx.StaticText(self, label=camera_name)
        font = title.GetFont()
        font.MakeBold()
        title.SetFont(font)

        vbox.Add(title, flag=wx.ALL | wx.ALIGN_CENTER_HORIZONTAL, border=5)

        # Some example buttons for this camera
        self.btn_connect = wx.Button(self, label="Connect")
        self.btn_disconnect = wx.Button(self, label="Disconnect")
        self.btn_start = wx.Button(self, label="Start Capture")
        self.btn_stop = wx.Button(self, label="Stop Capture")
        self.btn_snapshot = wx.Button(self, label="Snapshot")
        self.btn_settings = wx.Button(self, label="Settings")

        for btn in [
            self.btn_connect,
            self.btn_disconnect,
            self.btn_start,
            self.btn_stop,
            self.btn_snapshot,
            self.btn_settings,
        ]:
            vbox.Add(btn, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, border=5)

        # Spacer so panels stretch nicely
        vbox.AddStretchSpacer(1)

        self.SetSizer(vbox)

        # Example event binding (just to show how you'd hook logic)
        self.btn_connect.Bind(wx.EVT_BUTTON, self.on_connect_clicked)
        self.btn_disconnect.Bind(wx.EVT_BUTTON, self.on_disconnect_clicked)

    def on_connect_clicked(self, event):
        cam_label = self.GetCameraLabel()
        print(f"[{cam_label}] Connect pressed")

    def on_disconnect_clicked(self, event):
        cam_label = self.GetCameraLabel()
        print(f"[{cam_label}] Disconnect pressed")

    def GetCameraLabel(self):
        """
        Helper: read the title text for this panel.
        """
        # First child is the wx.StaticText we created.
        # Safer way is to search for first StaticText.
        for child in self.GetChildren():
            if isinstance(child, wx.StaticText):
                return child.GetLabel()
        return "Unknown Camera"


class SystemControl(wx.Frame):
    """
    Top-level frame that holds multiple CameraControl panels.
    """
    def __init__(self, parent=None, title="System Control"):
        super().__init__(parent, title=title, size=(1100, 450))

        # Outer panel so we can attach a sizer
        outer_panel = wx.Panel(self)

        # Horizontal layout to hold multiple CameraControl panels
        hbox = wx.BoxSizer(wx.HORIZONTAL)

        # Define the cameras that exist in the system
        camera_names = ["Camera A", "Camera B", "Camera C"]

        # Create one CameraControl panel per camera and add it to the row
        for cam_name in camera_names:
            cam_panel = CameraControl(outer_panel, camera_name=cam_name)

            # Put each camera panel in a StaticBoxSizer for visual grouping
            box = wx.StaticBox(outer_panel, label=cam_name + " Controls")
            static_sizer = wx.StaticBoxSizer(box, wx.VERTICAL)
            static_sizer.Add(cam_panel, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

            # Add to the main horizontal layout
            hbox.Add(static_sizer, proportion=1, flag=wx.EXPAND | wx.ALL, border=5)

        outer_panel.SetSizer(hbox)

        # Basic frame setup
        self.Centre()
        self.Show()


if __name__ == "__main__":
    app = wx.App()
    frame = SystemControl()
    app.MainLoop()
