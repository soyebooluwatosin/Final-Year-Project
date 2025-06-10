/**
 * Utility functions for working with the camera
 */

/**
 * Check if the device has a camera available
 * @returns Promise<boolean>
 */
export const checkCameraAvailability = async (): Promise<boolean> => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
      return false;
    }
  
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.some(device => device.kind === 'videoinput');
    } catch (error) {
      console.error('Error checking camera availability:', error);
      return false;
    }
  };
  
  /**
   * Test camera access to ensure permissions are granted
   * @returns Promise with result object containing success status and error if failed
   */
  export const testCameraAccess = async (): Promise<{ success: boolean; error?: string }> => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return { 
        success: false, 
        error: "Your browser doesn't support camera access"
      };
    }
  
    try {
      // Just attempt to get camera access
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      
      // Stop all tracks immediately after test
      stream.getTracks().forEach(track => track.stop());
      
      return { success: true };
    } catch (error) {
      console.error('Camera access test failed:', error);
      
      let errorMessage = "Failed to access camera";
      
      // More specific error messages based on the error type
      if (error instanceof DOMException) {
        switch (error.name) {
          case 'NotFoundError':
            errorMessage = "No camera found on your device";
            break;
          case 'NotAllowedError':
          case 'PermissionDeniedError':
            errorMessage = "Camera access denied. Please allow camera access in your browser permissions";
            break;
          case 'NotReadableError':
          case 'TrackStartError':
            errorMessage = "Camera is already in use by another application";
            break;
          case 'OverconstrainedError':
            errorMessage = "No camera matching the requested constraints was found";
            break;
          case 'TypeError':
            errorMessage = "No video track could be found";
            break;
          default:
            errorMessage = `Camera error: ${error.message}`;
        }
      }
      
      return { success: false, error: errorMessage };
    }
  };
  
  /**
   * Get ideal camera constraints based on the device
   * @param preferFrontCamera - Whether to prefer front facing camera
   * @returns MediaStreamConstraints object
   */
  export const getIdealCameraConstraints = (preferFrontCamera = true): MediaStreamConstraints => {
    // Basic constraints
    const constraints: MediaStreamConstraints = {
      video: {
        facingMode: preferFrontCamera ? "user" : "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    };
  
    // Mobile device detection (rough estimate)
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    
    if (isMobile) {
      // Adjust constraints for mobile devices
      const videoConstraints = constraints.video as MediaTrackConstraints;
      videoConstraints.width = { ideal: 720 };
      videoConstraints.height = { ideal: 1280 };
    }
  
    return constraints;
  };
  
  /**
   * Handles camera errors and returns user-friendly messages
   * @param error - The error object from getUserMedia
   * @returns User-friendly error message
   */
  export const getCameraErrorMessage = (error: unknown): string => {
    if (!error) {
      return "Unknown camera error occurred";
    }
    
    // Handle DOMException errors from getUserMedia
    if (error instanceof DOMException) {
      switch (error.name) {
        case 'NotFoundError':
          return "No camera was found on your device. Please connect a camera and try again.";
        case 'NotAllowedError':
        case 'PermissionDeniedError':
          return "Camera access was denied. Please allow camera access in your browser settings.";
        case 'NotReadableError':
        case 'TrackStartError':
          return "Camera is in use by another application. Please close other applications using your camera.";
        case 'OverconstrainedError':
          return "Your camera does not meet the required specifications.";
        case 'AbortError':
          return "Camera access was aborted. Please try again.";
        case 'TypeError':
          return "No video source is available.";
        default:
          return `Camera error: ${error.message}`;
      }
    }
    
    // Handle Error objects
    if (error instanceof Error) {
      return `Camera error: ${error.message}`;
    }
    
    // Handle unknown error types
    return "Failed to access camera. Please check your device settings.";
  };
  
  /**
   * Takes a screenshot from a video element
   * @param videoElement - The video element to capture from
   * @returns Base64 encoded image data
   */
  export const captureImageFromVideo = (videoElement: HTMLVideoElement): string => {
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error("Could not get canvas context for image capture");
    }
    
    // Draw the video frame to the canvas
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
    // Convert to base64
    try {
      return canvas.toDataURL('image/png');
    } catch (e) {
      console.error("Error converting canvas to data URL:", e);
      throw new Error("Failed to capture image from camera");
    }
  };