import { Button } from "antd"
import React, { useEffect, useRef, useState } from "react"

const CameraComponent: React.FC<{
  setCapturedImage: (image: string) => void
}> = ({ setCapturedImage }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false)
  const [cameraError, setCameraError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState<boolean>(true)
  const [videoState, setVideoState] = useState<string>("initializing")

  useEffect(() => {
    let mounted = true
    let stream: MediaStream | null = null

    const startCamera = async () => {
      console.log("Starting camera...")
      setVideoState("requesting permissions")

      try {
        if (!navigator.mediaDevices?.getUserMedia) {
          console.error("getUserMedia not supported")
          throw new Error("Camera not supported")
        }

        const constraints = {
          video: {
            facingMode: "user",
            width: { ideal: 640 },
            height: { ideal: 480 },
          },
        }

        console.log("Requesting getUserMedia with constraints:", constraints)
        stream = await navigator.mediaDevices.getUserMedia(constraints)

        if (!mounted) {
          stream.getTracks().forEach((track) => track.stop())
          return
        }

        console.log("Stream obtained:", stream)
        setVideoState("stream obtained")

        if (videoRef.current && stream) {
          console.log("Setting video src object")
          setVideoState("setting video source")

          // Clear any existing source first
          videoRef.current.srcObject = null

          // Set new source
          videoRef.current.srcObject = stream

          // Ensure the video element has correct attributes
          videoRef.current.autoplay = true
          videoRef.current.playsInline = true
          videoRef.current.muted = true

          videoRef.current.onplaying = () => {
            console.log("Video is playing")
            if (mounted) {
              setIsCameraActive(true)
              setIsLoading(false)
              setVideoState("playing")
              setCameraError(null) // Clear any previous errors
            }
          }

          videoRef.current.onloadedmetadata = () => {
            console.log("Video metadata loaded")
            setVideoState("metadata loaded")

            // Force play
            videoRef.current
              ?.play()
              .then(() => {
                console.log("Play successful")
                setVideoState("play initiated")
              })
              .catch((err: Error) => {
                console.error("Play failed:", err)
                setVideoState(`play failed: ${err.message}`)

                // Try clicking to play after a delay
                setTimeout(() => {
                  videoRef.current
                    ?.play()
                    .then(() => console.log("Second attempt successful"))
                    .catch((e: Error) =>
                      console.error("Second attempt failed:", e)
                    )
                }, 1000)
              })
          }

          videoRef.current.onerror = (e: string | Event) => {
            console.error("Video element error:", e)
            setVideoState("video error")
            if (mounted) {
              setCameraError("Video error occurred")
              setIsLoading(false)
            }
          }

          videoRef.current.onstalled = () => {
            console.log("Video stalled")
            setVideoState("stalled")
          }

          videoRef.current.onsuspend = () => {
            console.log("Video suspended")
            setVideoState("suspended")
          }
        }
      } catch (err: unknown) {
        console.error("Error accessing camera:", err)
        const error = err as Error
        if (mounted) {
          setVideoState(`error: ${error.message || "unknown"}`)
          setCameraError(error.message || "Unable to access camera")
          setIsCameraActive(false)
          setIsLoading(false)
        }
      }
    }

    startCamera()

    return () => {
      console.log("Cleaning up camera component")
      mounted = false
      if (stream) {
        stream.getTracks().forEach((track) => {
          console.log("Stopping track:", track.label)
          track.stop()
        })
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
    }
  }, [])

  const takePicture = () => {
    console.log("Take picture button clicked")

    if (!videoRef.current || !canvasRef.current) {
      console.log("Video or canvas not available")
      setCameraError("Video or canvas not ready")
      return
    }

    const video = videoRef.current
    const canvas = canvasRef.current

    console.log(
      "Taking picture - video dimensions:",
      video.videoWidth,
      "x",
      video.videoHeight
    )

    if (!video.videoWidth || !video.videoHeight) {
      console.error("Video has no dimensions")
      setCameraError("Video not ready for capture")
      return
    }

    try {
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight

      const ctx = canvas.getContext("2d")
      if (!ctx) {
        throw new Error("Could not get canvas context")
      }

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
      const imageData = canvas.toDataURL("image/png")

      console.log("Image captured successfully, data length:", imageData.length)
      console.log("Calling setCapturedImage with image data")

      // Call the parent function to handle the captured image
      setCapturedImage(imageData)
    } catch (error) {
      console.error("Error capturing image:", error)
      setCameraError("Failed to capture image")
    }
  }

  const forcePlay = () => {
    if (videoRef.current) {
      videoRef.current
        .play()
        .then(() => {
          console.log("Manual play successful")
          setIsCameraActive(true)
          setIsLoading(false)
          setCameraError(null)
        })
        .catch((err: Error) => {
          console.error("Manual play failed:", err)
          setCameraError("Failed to start video")
        })
    }
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div className="relative aspect-[3/2] w-full max-w-md overflow-hidden rounded-lg bg-black">
        {/* Video element */}
        <video
          ref={videoRef}
          className="h-full w-full object-cover"
          style={{
            width: "100%",
            height: "100%",
            display: "block",
            transform: "scaleX(-1)",
          }}
        />

        {/* Overlay for status */}
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          {isLoading && (
            <div className="bg-opacity-50 rounded bg-black p-4 text-center text-white">
              <div className="mx-auto mb-2 h-8 w-8 animate-spin rounded-full border-b-2 border-white"></div>
              <div className="text-sm">State: {videoState}</div>
            </div>
          )}
          {cameraError && (
            <div className="bg-opacity-90 max-w-xs rounded bg-red-500 p-4 text-center text-white">
              <div className="mb-2 font-semibold">Camera Error</div>
              <div className="text-sm">{cameraError}</div>
            </div>
          )}
        </div>

        {/* Status indicator */}
        <div className="bg-opacity-50 absolute top-2 left-2 rounded bg-black px-2 py-1 text-xs text-white">
          {isCameraActive ? "🔴 Live" : "⚫ Offline"}
        </div>

        {/* Debug info - remove in production */}
        <div className="bg-opacity-50 absolute bottom-2 left-2 max-w-xs rounded bg-black px-2 py-1 text-xs text-white">
          State: {videoState}
        </div>
      </div>

      <div className="flex gap-4">
        <Button
          type="primary"
          className="!rounded-lg !px-8 !py-5 !text-lg"
          onClick={takePicture}
          disabled={!isCameraActive}
        >
          Take Picture
        </Button>

        {!isCameraActive && (
          <Button
            type="default"
            className="!rounded-lg !px-8 !py-5 !text-lg"
            onClick={forcePlay}
          >
            Start Camera
          </Button>
        )}
      </div>

      {/* Hidden canvas */}
      <canvas ref={canvasRef} className="hidden" />
    </div>
  )
}

export default CameraComponent
