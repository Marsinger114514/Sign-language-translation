// pages/index/index.js
Page({
  data: {
    devicePosition: 'back',
    cameraReady: false,
    hasCameraAuth: false,
    fps: 1,
    autoMode: false,
    result: null,
    statusText: '',
    isProcessing: false,
    serverUrl: 'http://10.85.91.35:5000/api/predict/coordinates'
  },

  onLoad() {
    this.cameraContext = null;
    this.staticSession = null;
    this.staticSessionReady = null;
    this.pendingDetectTask = null;
    this.autoTimer = null;
    this.permissionPrompting = false;

    this.staticSessionReady = this.initStaticVisionKit().catch((err) => {
      console.error('initStaticVisionKit failed:', err);
      this.staticSessionReady = null;
      return null;
    });

    this.checkAndRequestCameraAuth();
  },

  onUnload() {
    this.stopAutoDetect();
    if (this.staticSession) {
      this.staticSession.stop();
      this.staticSession = null;
    }
  },

  checkAndRequestCameraAuth() {
    wx.getSetting({
      success: (res) => {
        if (res.authSetting['scope.camera']) {
          this.setData({ hasCameraAuth: true });
          this.initCamera();
          return;
        }

        wx.authorize({
          scope: 'scope.camera',
          success: () => {
            this.setData({ hasCameraAuth: true });
            this.initCamera();
          },
          fail: () => {
            this.setData({ hasCameraAuth: false, cameraReady: false });
            this.setStatus('未授予相机权限，请先开启权限');
            this.openCameraSettingModal();
          }
        });
      }
    });
  },

  openCameraSettingModal() {
    if (this.permissionPrompting) {
      return;
    }
    this.permissionPrompting = true;

    wx.showModal({
      title: '需要相机权限',
      content: '拍照与自动识别需要相机权限，是否去设置开启？',
      confirmText: '去设置',
      success: (modalRes) => {
        if (!modalRes.confirm) {
          this.setStatus('相机权限未开启');
          return;
        }

        wx.openSetting({
          success: (settingRes) => {
            if (settingRes.authSetting['scope.camera']) {
              this.setData({ hasCameraAuth: true, cameraReady: true });
              this.initCamera();
              this.setStatus('相机权限已开启');
            } else {
              this.setData({ hasCameraAuth: false, cameraReady: false });
              this.setStatus('相机权限仍未开启');
            }
          }
        });
      },
      complete: () => {
        this.permissionPrompting = false;
      }
    });
  },

  onOpenSetting(e) {
    const auth = (e && e.detail && e.detail.authSetting) || {};
    if (auth['scope.camera']) {
      this.setData({ hasCameraAuth: true, cameraReady: true });
      this.initCamera();
      this.setStatus('相机权限已开启');
      return;
    }

    this.setData({ hasCameraAuth: false, cameraReady: false });
    this.setStatus('相机权限未开启');
  },

  initCamera() {
    if (!this.data.hasCameraAuth) {
      return;
    }
    if (!this.cameraContext) {
      this.cameraContext = wx.createCameraContext();
    }
    this.setData({ cameraReady: true });
  },

  initStaticVisionKit() {
    return new Promise((resolve, reject) => {
      if (typeof wx.createVKSession !== 'function') {
        reject(new Error('当前微信版本不支持 VisionKit'));
        return;
      }

      const session = wx.createVKSession({
        track: { hand: { mode: 2 } }
      });

      session.on('error', (err) => {
        console.error('static VKSession error:', err);
      });

      session.on('updateAnchors', (anchors) => {
        if (!this.pendingDetectTask) {
          return;
        }
        const task = this.pendingDetectTask;
        this.pendingDetectTask = null;
        task.resolve(anchors || []);
      });

      session.start((errno) => {
        if (errno) {
          reject(new Error(`static session start failed: ${errno}`));
          return;
        }
        this.staticSession = session;
        this.setStatus('静态手势识别会话已就绪');
        resolve(session);
      });
    });
  },

  async ensureStaticSession() {
    if (this.staticSession) {
      return this.staticSession;
    }

    if (this.staticSessionReady) {
      const session = await this.staticSessionReady;
      if (session) {
        return session;
      }
    }

    this.staticSessionReady = this.initStaticVisionKit().catch((err) => {
      console.error('re-init static session failed:', err);
      this.staticSessionReady = null;
      return null;
    });

    return await this.staticSessionReady;
  },

  async chooseImage() {
    if (this.data.isProcessing) {
      this.setStatus('正在处理中');
      return;
    }

    const session = await this.ensureStaticSession();
    if (!session) {
      this.setStatus('识别会话未就绪');
      wx.showToast({ title: '识别会话未就绪', icon: 'none' });
      return;
    }

    wx.chooseMedia({
      count: 1,
      mediaType: ['image'],
      sourceType: ['album'],
      success: (res) => {
        const file = (res.tempFiles && res.tempFiles[0]) || null;
        const imagePath = file ? file.tempFilePath : '';
        this.recognizeFromImage(imagePath, '相册识别', { maxSide: 520 });
      },
      fail: () => {
        this.setStatus('选择图片失败');
      }
    });
  },

  async manualCapture() {
    if (!this.data.hasCameraAuth) {
      this.setStatus('未授予相机权限，请先开启权限');
      this.openCameraSettingModal();
      return;
    }

    if (!this.cameraContext || !this.data.cameraReady) {
      this.setStatus('相机未就绪');
      return;
    }

    if (this.data.isProcessing) {
      this.setStatus('正在处理中');
      return;
    }

    const session = await this.ensureStaticSession();
    if (!session) {
      this.setStatus('识别会话未就绪');
      return;
    }

    const frameRecognized = await this.recognizeFromCurrentFrame('拍照识别');
    if (frameRecognized) {
      return;
    }

    this.setStatus('拍照识别: 当前帧未检出，正在使用拍照回退...');
    try {
      const photoPath = await this.takePhotoPath('high');
      const normalizedPath = await this.normalizeCapturedImage(photoPath);
      this.recognizeFromImage(normalizedPath, '拍照识别', { maxSide: 720 });
    } catch (err) {
      console.error('takePhoto fallback failed:', err);
      this.setStatus('拍照失败');
    }
  },

  takePhotoPath(quality = 'high') {
    return new Promise((resolve, reject) => {
      if (!this.cameraContext) {
        reject(new Error('camera context not ready'));
        return;
      }

      this.cameraContext.takePhoto({
        quality,
        success: (res) => {
          const path = res && res.tempImagePath;
          if (!path) {
            reject(new Error('empty tempImagePath'));
            return;
          }
          resolve(path);
        },
        fail: (err) => reject(err || new Error('takePhoto failed'))
      });
    });
  },

  captureOneCameraFrame(timeoutMs = 1800) {
    return new Promise((resolve, reject) => {
      if (!this.cameraContext || typeof this.cameraContext.onCameraFrame !== 'function') {
        reject(new Error('onCameraFrame not supported'));
        return;
      }

      let done = false;
      let timer = null;
      let listener = null;

      const cleanup = () => {
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
        if (listener && typeof listener.stop === 'function') {
          try {
            listener.stop();
          } catch (e) {}
        }
      };

      listener = this.cameraContext.onCameraFrame((frame) => {
        if (done || !frame || !frame.data || !frame.width || !frame.height) {
          return;
        }

        done = true;
        cleanup();
        let frameBuffer = frame.data;
        if (frameBuffer instanceof ArrayBuffer) {
          frameBuffer = frameBuffer.slice(0);
        } else if (frameBuffer && frameBuffer.buffer instanceof ArrayBuffer) {
          frameBuffer = frameBuffer.buffer.slice(0);
        }

        resolve({
          frameBuffer,
          width: Number(frame.width) || 0,
          height: Number(frame.height) || 0
        });
      });

      try {
        listener.start();
      } catch (err) {
        done = true;
        cleanup();
        reject(err);
        return;
      }

      timer = setTimeout(() => {
        if (done) {
          return;
        }
        done = true;
        cleanup();
        reject(new Error('camera frame timeout'));
      }, timeoutMs);
    });
  },

  async recognizeFromCurrentFrame(source) {
    this.setData({ isProcessing: true });
    this.setStatus(`${source}: capturing current frame...`);

    try {
      const frame = await this.captureOneCameraFrame(1800);
      if (!frame.width || !frame.height || !frame.frameBuffer) {
        this.setStatus(`${source}: current frame invalid`);
        this.setData({ isProcessing: false });
        return false;
      }

      const landmarks = await this.detectLandmarksWithRetry(frame.frameBuffer, frame.width, frame.height);
      if (!landmarks) {
        this.setStatus(`${source}: no hand in current frame`);
        this.setData({ isProcessing: false });
        return false;
      }

      this.sendCoordinates(landmarks, source);
      return true;
    } catch (err) {
      console.warn('recognizeFromCurrentFrame failed:', err);
      this.setData({ isProcessing: false });
      return false;
    }
  },
  normalizeCapturedImage(imagePath) {
    return new Promise((resolve) => {
      if (!imagePath) {
        resolve('');
        return;
      }
      wx.getImageInfo({
        src: imagePath,
        success: (info) => {
          const isHuge = (info.width || 0) * (info.height || 0) > 12000000;
          if (!isHuge) {
            resolve(imagePath);
            return;
          }
          wx.compressImage({
            src: imagePath,
            quality: 90,
            success: (res) => resolve(res.tempFilePath || imagePath),
            fail: () => resolve(imagePath)
          });
        },
        fail: () => resolve(imagePath)
      });
    });
  },

  startAutoDetect() {
    this.stopAutoDetect();

    const interval = Math.max(1000 / Math.max(Number(this.data.fps) || 1, 0.2), 1200);
    this.autoTimer = setInterval(() => {
      if (!this.data.autoMode || this.data.isProcessing) {
        return;
      }
      this.manualCapture();
    }, interval);
  },

  stopAutoDetect() {
    if (!this.autoTimer) {
      return;
    }
    clearInterval(this.autoTimer);
    this.autoTimer = null;
  },

  async recognizeFromImage(imagePath, source, options = {}) {
    if (!imagePath) {
      this.setStatus(`${source}: 图片无效`);
      return;
    }

    this.setData({ isProcessing: true });
    this.setStatus(`${source}: 提取坐标中...`);

    try {
      const maxSide = Number(options.maxSide) || 480;
      const { frameBuffer, width, height } = await this.imageToFrameBuffer(imagePath, maxSide);
      const landmarks = await this.detectLandmarksWithRetry(frameBuffer, width, height);

      if (!landmarks) {
        this.setStatus(`${source}: 未检测到手势，请调整手势后重试`);
        wx.showToast({ title: '未检测到手势', icon: 'none' });
        this.setData({ isProcessing: false });
        return;
      }

      this.sendCoordinates(landmarks, source);
    } catch (err) {
      console.error('recognizeFromImage failed:', err);
      this.setStatus(`${source}: 坐标提取失败`);
      wx.showToast({ title: '识别失败，请重试', icon: 'none' });
      this.setData({ isProcessing: false });
    }
  },

  imageToFrameBuffer(imagePath, maxSide = 480) {
    return new Promise((resolve, reject) => {
      wx.getImageInfo({
        src: imagePath,
        success: (imgInfo) => {
          const orientation = String(imgInfo.orientation || 'up').toLowerCase();

          const baseScale = Math.max(imgInfo.width, imgInfo.height) > maxSide
            ? maxSide / Math.max(imgInfo.width, imgInfo.height)
            : 1;

          const drawW = Math.max(1, Math.round(imgInfo.width * baseScale));
          const drawH = Math.max(1, Math.round(imgInfo.height * baseScale));

          const rotateRight = orientation === 'right';
          const rotateLeft = orientation === 'left';
          const rotateDown = orientation === 'down';

          const outW = (rotateRight || rotateLeft) ? drawH : drawW;
          const outH = (rotateRight || rotateLeft) ? drawW : drawH;

          const drawAndResolve = (canvas, ctx) => {
            const img = canvas.createImage();
            img.onload = () => {
              ctx.clearRect(0, 0, outW, outH);
              ctx.save();

              if (rotateRight) {
                ctx.translate(outW, 0);
                ctx.rotate(Math.PI / 2);
              } else if (rotateLeft) {
                ctx.translate(0, outH);
                ctx.rotate(-Math.PI / 2);
              } else if (rotateDown) {
                ctx.translate(outW, outH);
                ctx.rotate(Math.PI);
              }

              ctx.drawImage(img, 0, 0, drawW, drawH);
              ctx.restore();

              const imageData = ctx.getImageData(0, 0, outW, outH);
              resolve({
                frameBuffer: imageData.data.buffer,
                width: outW,
                height: outH
              });
            };
            img.onerror = () => reject(new Error('图片加载失败'));
            img.src = imagePath;
          };

          if (typeof wx.createOffscreenCanvas === 'function') {
            const canvas = wx.createOffscreenCanvas({
              type: '2d',
              width: outW,
              height: outH
            });
            const ctx = canvas.getContext('2d');
            drawAndResolve(canvas, ctx);
            return;
          }

          const query = wx.createSelectorQuery();
          query.select('#bufferCanvas').fields({ node: true, size: true }).exec((res) => {
            const canvasNode = res && res[0] && res[0].node;
            if (!canvasNode) {
              reject(new Error('Canvas未找到'));
              return;
            }
            canvasNode.width = outW;
            canvasNode.height = outH;
            const ctx = canvasNode.getContext('2d');
            drawAndResolve(canvasNode, ctx);
          });
        },
        fail: () => reject(new Error('读取图片信息失败'))
      });
    });
  },

  async detectLandmarksWithRetry(frameBuffer, width, height) {
    const anchors = await this.detectHandOnce(frameBuffer, width, height, 0.12, 450);
    return this.extractLandmarks(anchors);
  },
  async detectHandOnce(frameBuffer, width, height, scoreThreshold, timeoutMs) {
    const session = await this.ensureStaticSession();
    if (!session) {
      return [];
    }

    return new Promise((resolve) => {
      if (this.pendingDetectTask) {
        this.pendingDetectTask.resolve([]);
        this.pendingDetectTask = null;
      }

      const timer = setTimeout(() => {
        if (!this.pendingDetectTask) {
          return;
        }
        this.pendingDetectTask = null;
        resolve([]);
      }, timeoutMs);

      this.pendingDetectTask = {
        resolve: (anchors) => {
          clearTimeout(timer);
          resolve(anchors || []);
        }
      };

      try {
        session.detectHand({
          frameBuffer,
          width,
          height,
          scoreThreshold,
          algoMode: 2
        });
      } catch (err) {
        clearTimeout(timer);
        this.pendingDetectTask = null;
        resolve([]);
      }
    });
  },

  extractLandmarks(anchors) {
    if (!anchors || !anchors.length) {
      return null;
    }

    const hand = anchors[0] || {};
    const points = hand.points || hand.keyPoints || hand.landmarks || [];
    if (points.length < 21) {
      return null;
    }

    return points.slice(0, 21).map((p) => {
      if (Array.isArray(p)) {
        return [Number(p[0]) || 0, Number(p[1]) || 0, Number(p[2]) || 0];
      }
      return [Number(p.x) || 0, Number(p.y) || 0, Number(p.z) || 0];
    });
  },

  sendCoordinates(landmarks, source) {
    this.setStatus(`${source}: 检测到 ${landmarks.length} 点，上传中...`);

    wx.request({
      url: this.data.serverUrl,
      method: 'POST',
      header: { 'content-type': 'application/json' },
      data: {
        landmarks,
        source,
        timestamp: Date.now()
      },
      timeout: 8000,
      success: (res) => {
        const data = res.data || {};
        if (res.statusCode === 200 && data.letter && data.letter !== '错误') {
          this.setData({
            result: {
              letter: data.letter,
              confidencePercent: ((Number(data.confidence) || 0) * 100).toFixed(1),
              source
            }
          });
          this.setStatus(`${source}: 识别成功 ${data.letter}`);
        } else {
          this.setStatus(`${source}: 服务器返回 ${data.error || '识别失败'}`);
        }
      },
      fail: (err) => {
        console.error('sendCoordinates failed:', err);
        this.setStatus(`${source}: 网络错误`);
      },
      complete: () => {
        this.setData({ isProcessing: false });
      }
    });
  },

  switchCamera() {
    const devicePosition = this.data.devicePosition === 'back' ? 'front' : 'back';
    this.setData({ devicePosition });
    this.setStatus(`已切换到${devicePosition === 'back' ? '后置' : '前置'}摄像头`);
  },

  onFpsChange(e) {
    const fps = Number(e.detail.value) || 1;
    this.setData({ fps });
    this.setStatus(`识别频率 ${fps} 次/秒`);

    if (this.data.autoMode) {
      this.startAutoDetect();
    }
  },

  toggleAutoMode(e) {
    const autoMode = !!e.detail.value;

    if (autoMode && !this.data.hasCameraAuth) {
      this.setData({ autoMode: false });
      this.setStatus('请先开启相机权限');
      this.openCameraSettingModal();
      return;
    }

    this.setData({ autoMode });
    if (autoMode) {
      this.startAutoDetect();
      this.setStatus('自动识别已开启（静态检测）');
    } else {
      this.stopAutoDetect();
      this.setStatus('自动识别已关闭');
    }
  },

  onCameraInit() {
    this.setData({ cameraReady: true });
    this.setStatus('相机已就绪');
  },

  onCameraError(e) {
    const detail = (e && e.detail) || {};
    const msg = String(detail.msg || detail.errMsg || '');

    console.error('camera error:', detail);
    this.setData({ cameraReady: false });

    if (msg.includes('user cancel auth') || msg.includes('auth deny') || msg.includes('auth denied')) {
      this.setData({ hasCameraAuth: false, autoMode: false });
      this.stopAutoDetect();
      this.setStatus('相机权限被拒绝，请在设置中开启');
      this.openCameraSettingModal();
      return;
    }

    this.setStatus('相机错误');
  },

  setStatus(text) {
    this.setData({ statusText: text || '' });
  }
});



