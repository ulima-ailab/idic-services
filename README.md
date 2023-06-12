# idic-services

This is the "idic-services" project, a Django-based web application for providing services to customers. It allows users to request and manage various services offered by the company.

## Getting Started

To run the "idic-services" project locally, follow these steps:

### Prerequisites

- Python 3.x
- pip package manager

### Installation

1. Clone the repository:
   ```shell
   git clone https://github.com/your-username/idic-services.git

2. Change to the project directory:
    ```shell
    cd idic-services

3. Create a virtual environment (optional but recommended):
    ```shell
    python3 -m venv env
    source env/bin/activate

4. Install the project dependencies:
    ```shell
    pip install -r requirements.txt

5. Apply the database migrations:
    ```shell
    python manage.py migrate

6. Create a superuser (admin) account:
    ```shell
    python manage.py createsuperuser


## Usage

To run the "idic-services" project locally, follow these steps:

1. Start the development server:
   ```shell
   python manage.py runserver

2. Open a web browser and visit http://127.0.0.1:8000/ to access the project.

3. Log in with the superuser account created earlier to access the admin interface.

4. Explore the available services and perform necessary operations.


## Example
POST http://127.0.0.1:8000/get-emotions/

#### Request Parameters

| Parameter | Type | Required | Description |
| --- | --- | --- | --- |
| startDate | string | Yes | The start date of the range to get emotions for in the format `YYYY-MM-DD HH:mm:ss`. |
| endDate | string | Yes | The end date of the range to get emotions for in the format `YYYY-MM-DD HH:mm:ss`. |

startDate:2023-01-01 00:00:00
endDate:2024-07-31 00:00:00
