import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import axios from 'axios';
import { FaUpload, FaLeaf, FaExclamationTriangle, FaCheckCircle } from 'react-icons/fa';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [apiStatus, setApiStatus] = useState('checking');

  // Vérifier le statut de l'API au démarrage
  React.useEffect(() => {
    checkApiStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await axios.get('/ping');
      setApiStatus('connected');
    } catch (err) {
      setApiStatus('disconnected');
    }
  };

  const onDrop = useCallback((acceptedFiles) => {
    const uploadedFile = acceptedFiles[0];
    if (uploadedFile) {
      setFile(uploadedFile);
      setPreview(URL.createObjectURL(uploadedFile));
      setPrediction(null);
      setError(null);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpeg', '.jpg', '.png', '.gif']
    },
    multiple: false
  });

  const analyzeImage = async () => {
    if (!file) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000,
      });

      setPrediction(response.data);
    } catch (err) {
      console.error('Erreur lors de l\'analyse:', err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else if (err.code === 'ECONNABORTED') {
        setError('Délai d\'attente dépassé. Veuillez réessayer.');
      } else {
        setError('Erreur lors de l\'analyse de l\'image. Veuillez réessayer.');
      }
    } finally {
      setLoading(false);
    }
  };

  const getHealthStatusIcon = (status) => {
    switch (status) {
      case 'saine':
        return <FaCheckCircle className="status-icon healthy" />;
      case 'malade':
        return <FaExclamationTriangle className="status-icon sick" />;
      default:
        return <FaLeaf className="status-icon unknown" />;
    }
  };

  const getHealthStatusColor = (status) => {
    switch (status) {
      case 'saine':
        return '#4CAF50';
      case 'malade':
        return '#F44336';
      default:
        return '#FF9800';
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1>
            <FaLeaf className="header-icon" />
            Diagnostic Maladies Pommes de Terre
          </h1>
          <p>Analysez vos feuilles de pommes de terre avec l'IA</p>
        </header>

        {/* Statut de l'API */}
        <div className="card">
          <h3>Statut du système</h3>
          <div className={`api-status ${apiStatus}`}>
            {apiStatus === 'checking' && <div className="spinner"></div>}
            {apiStatus === 'connected' && (
              <div className="status-message">
                <FaCheckCircle className="status-icon" />
                API connectée - Prêt pour l'analyse
              </div>
            )}
            {apiStatus === 'disconnected' && (
              <div className="status-message">
                <FaExclamationTriangle className="status-icon" />
                API non connectée - Démarrez le serveur backend
              </div>
            )}
          </div>
        </div>

        {/* Zone de téléchargement */}
        <div className="card">
          <h3>Télécharger une image</h3>
          <div
            {...getRootProps()}
            className={`dropzone ${isDragActive ? 'active' : ''}`}
          >
            <input {...getInputProps()} />
            <FaUpload className="upload-icon" />
            {isDragActive ? (
              <p>Déposez l'image ici...</p>
            ) : (
              <p>Glissez-déposez une image ici, ou cliquez pour sélectionner</p>
            )}
          </div>
        </div>

        {/* Aperçu de l'image */}
        {preview && (
          <div className="card">
            <h3>Aperçu de l'image</h3>
            <div className="image-preview">
              <img src={preview} alt="Aperçu" />
              <button 
                className="btn analyze-btn" 
                onClick={analyzeImage}
                disabled={loading || apiStatus !== 'connected'}
              >
                {loading ? 'Analyse en cours...' : 'Analyser l\'image'}
              </button>
            </div>
          </div>
        )}

        {/* Résultats */}
        {prediction && (
          <div className="card">
            <h3>Résultats de l'analyse</h3>
            <div className="prediction-results">
              <div className="prediction-header">
                {getHealthStatusIcon(prediction.health_status)}
                <div>
                  <h4 style={{ color: getHealthStatusColor(prediction.health_status) }}>
                    {prediction.predicted_class.replace(/_/g, ' ')}
                  </h4>
                  <p>Confiance: {(prediction.confidence * 100).toFixed(1)}%</p>
                </div>
              </div>
              
              <div className="recommendation">
                <h5>Recommandation:</h5>
                <p>{prediction.recommendation}</p>
              </div>

              <div className="probabilities">
                <h5>Probabilités par classe:</h5>
                <div className="probability-bars">
                  {Object.entries(prediction.probabilities).map(([className, prob]) => (
                    <div key={className} className="probability-item">
                      <span className="class-name">
                        {className.replace(/_/g, ' ')}
                      </span>
                      <div className="probability-bar">
                        <div 
                          className="probability-fill"
                          style={{ width: `${prob * 100}%` }}
                        ></div>
                      </div>
                      <span className="probability-value">
                        {(prob * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Erreur */}
        {error && (
          <div className="card error-card">
            <h3>Erreur</h3>
            <p className="error-message">{error}</p>
          </div>
        )}

        {/* Instructions */}
        <div className="card">
          <h3>Instructions</h3>
          <ul>
            <li>Téléchargez une image claire d'une feuille de pomme de terre</li>
            <li>L'image doit être au format JPG, PNG ou GIF</li>
            <li>Assurez-vous que la feuille est bien visible et éclairée</li>
            <li>Le système détectera automatiquement les maladies</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default App; 