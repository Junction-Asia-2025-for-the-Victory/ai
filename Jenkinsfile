pipeline {
    agent any
    options {
        timestamps()
    }
    parameters {
        string(name: 'TAG', defaultValue: 'latest', description: 'Docker image tag')
    }
    environment {
        DOCKER_IMAGE = "victory-ai:${params.TAG}"
    }
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build Docker Image') {
            steps {
                sh """
                    echo "=== Building AI Server Docker Image ==="
                    docker build -t ${DOCKER_IMAGE} .
                    docker images | grep victory-ai
                """
            }
        }
    }
    post {
        success {
            echo "✅ AI Server image built successfully: ${DOCKER_IMAGE}"
        }
        failure {
            echo "❌ AI Server build failed"
        }
    }
}