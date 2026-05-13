import { useEffect, useRef, useState } from "react";
import { ContentSummarizationSection } from "./ContentSummarizationSection";
import { API_BASE_URL } from "./config";
import "./App.css";

type AppTab = "document" | "summarization";

function App() {
  const [activeTab, setActiveTab] = useState<AppTab>("document");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const [documentResult, setDocumentResult] = useState<any>(null);
  const [imageDescriptionResult, setImageDescriptionResult] = useState<any>(null);

  const [documentLoading, setDocumentLoading] = useState(false);
  const [descriptionLoading, setDescriptionLoading] = useState(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraReady, setCameraReady] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const DOCUMENT_API_URL = `${API_BASE_URL}/document/predict-document-type`;

  const IMAGE_DESCRIPTION_API_URL = `${API_BASE_URL}/image/describe-image`;

  useEffect(() => {
    if (!cameraActive) return;

    const openCamera = async () => {
      setCameraError(null);
      setCameraReady(false);

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        });

        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;

          videoRef.current.onloadedmetadata = async () => {
            try {
              await videoRef.current?.play();
              setCameraReady(true);
            } catch (error) {
              console.error(error);
              setCameraError("Camera preview failed to start.");
            }
          };
        }
      } catch (error) {
        console.error(error);
        setCameraError(
          "Camera access failed. Please allow camera permission or close other apps using the camera."
        );
        setCameraActive(false);
      }
    };

    openCamera();

    return () => {
      stopCamera();
    };
  }, [cameraActive]);

  const resizeImageBeforeUpload = (file: File): Promise<File> => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const reader = new FileReader();

      reader.onload = (event) => {
        img.src = event.target?.result as string;
      };

      img.onload = () => {
        const maxSize = 900;

        let width = img.width;
        let height = img.height;

        if (width > height) {
          if (width > maxSize) {
            height = Math.round((height * maxSize) / width);
            width = maxSize;
          }
        } else {
          if (height > maxSize) {
            width = Math.round((width * maxSize) / height);
            height = maxSize;
          }
        }

        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");

        if (!ctx) {
          reject(new Error("Could not resize image"));
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject(new Error("Could not create resized image"));
              return;
            }

            const resizedFile = new File([blob], "resized_document.jpg", {
              type: "image/jpeg",
            });

            resolve(resizedFile);
          },
          "image/jpeg",
          0.85
        );
      };

      img.onerror = () => {
        reject(new Error("Could not load image"));
      };

      reader.onerror = () => {
        reject(new Error("Could not read image file"));
      };

      reader.readAsDataURL(file);
    });
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];

    if (!file) return;

    stopCamera();

    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setDocumentResult(null);
    setImageDescriptionResult(null);
  };

  const startCamera = () => {
    setPreviewUrl(null);
    setSelectedFile(null);
    setDocumentResult(null);
    setImageDescriptionResult(null);
    setCameraError(null);
    setCameraReady(false);
    setCameraActive(true);
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setCameraReady(false);
    setCameraActive(false);
  };

  const capturePhoto = () => {
    if (!videoRef.current) {
      alert("Camera is not ready.");
      return;
    }

    const video = videoRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) {
      alert("Camera is still loading. Please wait a few seconds and try again.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      alert("Could not capture image.");
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    const dataUrl = canvas.toDataURL("image/jpeg", 0.95);

    fetch(dataUrl)
      .then((res) => res.blob())
      .then((blob) => {
        const file = new File([blob], "captured_document.jpg", {
          type: "image/jpeg",
        });

        setSelectedFile(file);
        setPreviewUrl(dataUrl);
        setDocumentResult(null);
        setImageDescriptionResult(null);

        stopCamera();
      })
      .catch((error) => {
        console.error(error);
        alert("Could not create captured image.");
      });
  };

  const predictDocument = async () => {
    if (!selectedFile) {
      alert("Please select or capture an image first.");
      return;
    }

    setDocumentLoading(true);
    setDocumentResult(null);

    try {
      const resizedFile = await resizeImageBeforeUpload(selectedFile);

      const formData = new FormData();
      formData.append("file", resizedFile);

      const response = await fetch(DOCUMENT_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend returned an error.");
      }

      const data = await response.json();
      setDocumentResult(data);
    } catch (error) {
      console.error(error);
      alert("Document prediction failed. Please check backend is running.");
    } finally {
      setDocumentLoading(false);
    }
  };

  const describeImage = async () => {
    if (!selectedFile) {
      alert("Please select or capture an image first.");
      return;
    }

    setDescriptionLoading(true);
    setImageDescriptionResult(null);

    try {
      const resizedFile = await resizeImageBeforeUpload(selectedFile);

      const formData = new FormData();
      formData.append("file", resizedFile);

      const response = await fetch(IMAGE_DESCRIPTION_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend returned an error.");
      }

      const data = await response.json();
      setImageDescriptionResult(data);
    } catch (error) {
      console.error(error);
      alert("Image description failed. Please check backend and OpenAI API key.");
    } finally {
      setDescriptionLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="container">
        <header className="header">
          <h1>Smart Wearable Reading Assistant</h1>
          <p>Document identification, image description, and content summarization</p>
        </header>

        <nav className="app-tabs" aria-label="Main modules">
          <button
            type="button"
            className={activeTab === "document" ? "app-tab active" : "app-tab"}
            onClick={() => setActiveTab("document")}
          >
            Document &amp; image
          </button>
          <button
            type="button"
            className={activeTab === "summarization" ? "app-tab active" : "app-tab"}
            onClick={() => setActiveTab("summarization")}
          >
            Summarization &amp; categories
          </button>
        </nav>

        {activeTab === "summarization" ? (
          <ContentSummarizationSection />
        ) : null}

        {activeTab === "document" ? (
        <div className="grid">
          <section className="card upload-card">
            <h2>Upload or Capture Page</h2>

            <p className="small-text">
              Upload a printed document image or inside page image.
            </p>

            <div className="action-row">
              <label className="file-label">
                Browse Image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden-input"
                />
              </label>

              <button
                type="button"
                onClick={cameraActive ? stopCamera : startCamera}
                className="camera-btn"
              >
                {cameraActive ? "Close Camera" : "Open Camera"}
              </button>
            </div>

            {cameraError && <p className="error-text">{cameraError}</p>}

            {cameraActive && (
              <div className="camera-box">
                <video
                  ref={videoRef}
                  autoPlay
                  playsInline
                  muted
                  className="camera-preview"
                />

                <p className="small-text">
                  {cameraReady
                    ? "Camera ready. Place the document clearly and capture."
                    : "Camera loading... please wait."}
                </p>

                <button
                  type="button"
                  onClick={capturePhoto}
                  className="capture-btn"
                  disabled={!cameraReady}
                >
                  Capture Photo
                </button>
              </div>
            )}

            {previewUrl && (
              <div className="preview-box">
                <img src={previewUrl} alt="Selected document" />
              </div>
            )}

            <button
              onClick={predictDocument}
              disabled={documentLoading || descriptionLoading}
              className="predict-btn"
            >
              {documentLoading ? "Predicting..." : "Predict Document"}
            </button>

            <button
              onClick={describeImage}
              disabled={documentLoading || descriptionLoading}
              className="describe-btn"
            >
              {descriptionLoading ? "Describing..." : "Describe Image"}
            </button>
          </section>

          <section className="card result-card">
            <h2>Model Output</h2>

            {!documentResult && !imageDescriptionResult && (
              <p className="empty-text">
                Results will appear here after prediction or image description.
              </p>
            )}

            {documentResult && (
              <div className="result-content">
                <h3 className="section-title">Document Identification Result</h3>

                <div className="result-row">
                  <span>Document Type</span>
                  <strong>{documentResult.document_type}</strong>
                </div>

                <div className="result-row">
                  <span>Confidence</span>
                  <strong>{documentResult.confidence}%</strong>
                </div>

                <div className="result-row">
                  <span>Detected Title</span>
                  <strong>
                    {documentResult.title ? documentResult.title : "Not needed"}
                  </strong>
                </div>

                <div className="message-box">
                  <h3>Final Device Message</h3>
                  <p>{documentResult.final_message}</p>
                </div>

                <div className="predictions">
                  <h3>Top Predictions</h3>

                  {documentResult.all_predictions?.map(
                    (item: any, index: number) => (
                      <div className="prediction-item" key={index}>
                        <span>{item.class_name}</span>
                        <span>{item.confidence}%</span>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {imageDescriptionResult && (
              <div className="result-content image-description-output">
                <h3 className="section-title">Image Description Result</h3>

                <div className="message-box description-box">
                  <h3>Simple Description</h3>
                  <p>{imageDescriptionResult.description}</p>
                </div>

                <div className="message-box">
                  <h3>Final Device Message</h3>
                  <p>{imageDescriptionResult.final_message}</p>
                </div>
              </div>
            )}
          </section>
        </div>
        ) : null}
      </div>
    </div>
  );
}

export default App;