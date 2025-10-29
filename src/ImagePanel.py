import queue
import wx
import os
from pathlib import Path
from skimage.feature import graycomatrix, graycoprops
from scipy.optimize import curve_fit
import numpy as np
import cv2
import datetime
import time
import threading
from pypylon import pylon
import wx.lib.agw.floatspin as FS
import csv
from VideoRecordingSession import VideoRecordingSession
from InputEventHandler import ConfigurationEventPrinter

class ImagePanel(wx.Panel):
    def __init__(self, parent, min_w=640, min_h=480):
        super().__init__(parent)
        self.bitmap = wx.Bitmap(min_w, min_h)   # placeholder
        self.SetBackgroundStyle(wx.BG_STYLE_PAINT)  # proper buffered painting
        self.SetDoubleBuffered(True)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self._on_size)
        # Optional baseline so sizers give it room:
        # self.SetMinSize((min_w, min_h))

    def _on_size(self, evt):
        self.Refresh(False)    # schedule repaint with new size
        evt.Skip()

    def OnPaint(self, evt):
        dc = wx.AutoBufferedPaintDCFactory(self)  # buffered DC
        dc.Clear()
        if not self.bitmap.IsOk():
            return
        pw, ph = self.GetClientSize()
        if pw <= 0 or ph <= 0:
            return

        bw, bh = self.bitmap.GetWidth(), self.bitmap.GetHeight()
        if bw != pw or bh != ph:
            bmp = wx.Bitmap(self.bitmap.ConvertToImage().Scale(pw, ph, wx.IMAGE_QUALITY_HIGH))
        else:
            bmp = self.bitmap
        dc.DrawBitmap(bmp, 0, 0, True)

    def update_bitmap(self, bmp: wx.Bitmap):
        self.bitmap = bmp
        self.Refresh(False)