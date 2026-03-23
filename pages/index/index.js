// pages/index/index.js
Page({
    data: {
      devicePosition: 'back',
      fps: 1,                     // 默认每秒1次
      autoMode: false,            // 是否开启自动拍照模式
      result: null,
      cameraContext: null,
      autoTimer: null,
      serverUrl: 'http://10.85.91.35:5000/api/predict/letter'  // 替换为实际地址
    },
  
    onLoad() {
      this.setData({
        cameraContext: wx.createCameraContext()
      });
    },
  
    onCameraInit() {
      console.log('✅ 相机初始化完成');
    },
  
    // 切换摄像头
    switchCamera() {
      this.setData({
        devicePosition: this.data.devicePosition === 'back' ? 'front' : 'back'
      });
    },
  
    // 从相册选择
    chooseImage() {
      wx.chooseImage({
        count: 1,
        sizeType: ['compressed'],
        sourceType: ['album'],
        success: (res) => {
          const tempFilePath = res.tempFilePaths[0];
          this.sendPhoto(tempFilePath);
        }
      });
    },
  
    // 手动拍照
    manualCapture() {
      this.takePhotoAndRecognize();
    },
  
    // 拍照并识别（核心函数）
    takePhotoAndRecognize() {
      if (!this.data.cameraContext) {
        wx.showToast({ title: '相机未就绪', icon: 'none' });
        return;
      }
      this.data.cameraContext.takePhoto({
        quality: 'high',
        success: (res) => {
          console.log('拍照成功:', res.tempImagePath);
          this.sendPhoto(res.tempImagePath);
        },
        fail: (err) => {
          console.error('拍照失败', err);
          wx.showToast({ title: '拍照失败', icon: 'none' });
        }
      });
    },
  
    // 发送照片到服务器
    sendPhoto(filePath) {
      wx.getFileSystemManager().readFile({
        filePath: filePath,
        encoding: 'base64',
        success: (fileRes) => {
          const base64 = fileRes.data;
          console.log('📦 图片 base64 长度:', base64.length);
          if (base64.length < 500) {
            console.warn('图片可能无效');
            return;
          }
          this.requestServer(base64);
        },
        fail: (err) => {
          console.error('读取文件失败', err);
        }
      });
    },
  
    // 请求服务器识别
    requestServer(base64Image) {
      wx.request({
        url: this.data.serverUrl,
        method: 'POST',
        data: { image: base64Image },
        success: (res) => {
          console.log('服务器返回:', res.data);
          if (res.statusCode === 200) {
            const data = res.data;
            const confidence = data.confidence || 0;
            const confidencePercent = (confidence * 100).toFixed(1);
            this.setData({
              result: {
                letter: data.letter || '未知',
                confidence: confidence,
                confidencePercent: confidencePercent
              }
            });
          } else {
            wx.showToast({ title: '识别失败', icon: 'none' });
          }
        },
        fail: (err) => {
          console.error('请求失败', err);
          wx.showToast({ title: '网络错误', icon: 'none' });
        }
      });
    },
  
    // 自动模式开关
    toggleAutoMode(e) {
      const autoMode = e.detail.value;
      this.setData({ autoMode });
      if (autoMode) {
        this.startAutoCapture();
      } else {
        this.stopAutoCapture();
      }
    },
  
    // 启动自动拍照
    startAutoCapture() {
      if (this.data.autoTimer) clearInterval(this.data.autoTimer);
      const interval = 1000 / this.data.fps;
      this.data.autoTimer = setInterval(() => {
        this.takePhotoAndRecognize();
      }, interval);
      console.log('自动拍照已启动，间隔:', interval, 'ms');
    },
  
    // 停止自动拍照
    stopAutoCapture() {
      if (this.data.autoTimer) {
        clearInterval(this.data.autoTimer);
        this.data.autoTimer = null;
        console.log('自动拍照已停止');
      }
    },
  
    // 频率变化
    onFpsChange(e) {
      const newFps = e.detail.value;
      this.setData({ fps: newFps });
      if (this.data.autoMode) {
        this.startAutoCapture(); // 重新设定定时器
      }
    },
  
    onUnload() {
      this.stopAutoCapture();
    },
  
    onCameraError(e) {
      console.error('相机错误', e.detail);
      wx.showToast({ title: '相机启动失败', icon: 'none' });
    }
  });