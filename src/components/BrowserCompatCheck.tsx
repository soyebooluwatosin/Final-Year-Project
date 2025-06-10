import { Alert, Typography } from "antd"
import { useEffect, useState } from "react"

const { Paragraph, Text, Link } = Typography

interface CompatibilityStatus {
  browserSupported: boolean
  cameraSupported: boolean
  browserName: string
  issues: string[]
}

/**
 * Component to check browser compatibility for camera features
 */
const BrowserCompatCheck: React.FC = () => {
  const [compatibility, setCompatibility] = useState<CompatibilityStatus>({
    browserSupported: true,
    cameraSupported: true,
    browserName: "Unknown",
    issues: [],
  })

  useEffect(() => {
    // Check browser compatibility
    const checkCompatibility = () => {
      const issues: string[] = []
      const userAgent = navigator.userAgent

      // Detect browser
      let browserName = "Unknown"
      if (userAgent.indexOf("Chrome") > -1) {
        browserName = "Chrome"
      } else if (userAgent.indexOf("Firefox") > -1) {
        browserName = "Firefox"
      } else if (userAgent.indexOf("Safari") > -1) {
        browserName = "Safari"
      } else if (
        userAgent.indexOf("Edge") > -1 ||
        userAgent.indexOf("Edg") > -1
      ) {
        browserName = "Edge"
      } else if (
        userAgent.indexOf("MSIE") > -1 ||
        userAgent.indexOf("Trident") > -1
      ) {
        browserName = "Internet Explorer"
        issues.push(
          "Internet Explorer is not supported. Please use Chrome, Firefox, or Edge."
        )
      }

      // Check for camera API support
      const cameraSupported = !!(
        navigator.mediaDevices && navigator.mediaDevices.getUserMedia
      )
      if (!cameraSupported) {
        issues.push(
          "Your browser doesn't support camera access. Please use Chrome, Firefox, or Edge."
        )
      }

      // Check for secure context (required for camera access in modern browsers)
      if (window.isSecureContext === false) {
        issues.push(
          "This page is not running in a secure context (HTTPS). Camera access may be blocked."
        )
      }

      // Check for private browsing mode in Safari
      if (browserName === "Safari") {
        try {
          const testKey = "test"
          localStorage.setItem(testKey, testKey)
          localStorage.removeItem(testKey)
        } catch (e) {
          issues.push(
            "Safari in Private Browsing Mode may block camera access. Please use regular browsing mode."
          )
        }
      }

      setCompatibility({
        browserSupported: browserName !== "Internet Explorer",
        cameraSupported,
        browserName,
        issues,
      })
    }

    checkCompatibility()
  }, [])

  // Only show if there are compatibility issues
  if (compatibility.issues.length === 0) {
    return null
  }

  return (
    <Alert
      type="warning"
      showIcon
      message="Browser Compatibility Warning"
      description={
        <div className="space-y-2">
          <Paragraph>
            We detected potential issues that might affect the camera
            functionality:
          </Paragraph>
          <ul className="list-disc pl-5">
            {compatibility.issues.map((issue, index) => (
              <li key={index}>{issue}</li>
            ))}
          </ul>
          <Paragraph>
            <Text strong>Current browser:</Text> {compatibility.browserName}
          </Paragraph>
          <Paragraph>
            For the best experience, we recommend using the latest version of{" "}
            <Link href="https://www.google.com/chrome/" target="_blank">
              Chrome
            </Link>
            ,{" "}
            <Link href="https://www.mozilla.org/firefox/" target="_blank">
              Firefox
            </Link>
            , or{" "}
            <Link href="https://www.microsoft.com/edge" target="_blank">
              Edge
            </Link>
            .
          </Paragraph>
        </div>
      }
    />
  )
}

export default BrowserCompatCheck
