## Running the Project

### Option 1: Using Docker
1. Install Docker on your machine
2. Pull the image: `docker pull ankitbourasi0/face-detection:latest`
3. Run the container: `docker run -p 8000:8000 ankitbourasi0/face-detection:latest`
4. Access the application at `http://localhost:8000`
5. Swagger access `http://localhost:8000/docs`

### Option 2: Local Setup
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run the application: `python app.py`
6. Access the application at `http://localhost:8000`

### Option 3: GitHub Codespaces
1. Click the 'Code' button on this repository
2. Select 'Open with Codespaces'
3. Follow the on-screen instructions to start your development environment
