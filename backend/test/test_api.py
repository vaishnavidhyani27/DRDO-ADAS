import base64
import requests


IMAGE_PATH = "test_images/car.jpg"
API_URL = "http://127.0.0.1:5000/detect"


def main():
    try:
        with open(IMAGE_PATH, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "image": f"data:image/jpeg;base64,{encoded_image}"
        }

        response = requests.post(
            API_URL,
            json=payload,
            timeout=120
        )

        print("Status code:", response.status_code)
        print("Response:")
        print(response.json())

    except FileNotFoundError:
        print(f"Image not found: {IMAGE_PATH}")

    except requests.RequestException as error:
        print("API request failed:", error)

    except Exception as error:
        print("Unexpected error:", error)


if __name__ == "__main__":
    main()