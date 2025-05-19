#include <stdio.h>
using namespace std;


#include <iostream>

// OpenCV includes
#include <opencv2/opencv.hpp>

//#include "opencv2/core.hpp"
//#include "opencv2/highgui.hpp"
using namespace cv;

//utworzenie okna o nazwie name w punkcie (x,y) (lewy górny róg okna)
void CreateWindowAt(const char* name, int x, int y)
{ 
	namedWindow(name, WINDOW_AUTOSIZE);
	moveWindow(name, x, y);
}
// wyświetlenie obrazu img w oknie o nazwie name położonego w punkcie (x,y) (lewy górny róg okna)
void ShowImageAt(const char* name, Mat img, int x, int y)
{ 
	CreateWindowAt(name, x, y);
	imshow(name, img);
}
// wczytanie obrazu z pliku name do macierzy img
int read_image(const char* name, Mat* img)
{
	*img = imread(name);
	if (!(*img).data)
	{
		cout << "Error! Cannot read source file. Press ENTER.";
		waitKey(); // czekaj na naciśnięcie klawisza
		return(-1);
	}
}

Mat srcImage;			// obraz wejściowy
Mat greyImage;			// obraz po konwersji do obrazu w odcieniach szarości


//funkcja konwertująca obraz src na obraz dst w odcieniach szarości
void convertToGrey(Mat src, Mat dst)
{
	//pętla po wszystkich pikselach obrazu
	for (int x = 0; x < src.cols; x++)
		for (int y = 0; y < src.rows; y++)
		{
			//pobranie do zmiennej pixelColor wszystkich 3 składowych koloru piksela
			Vec3b pixelColor = src.at<Vec3b>(y, x);
			//konwersja na kolor szary; pixelColor[0] składowa B, pixelColor[1] składowa G, pixelColor[2] składowa R
			int gray = (int)(0.299f * pixelColor[2] + 0.587f * pixelColor[1] + 0.114f * pixelColor[0]);
			for (int i = 0; i < 3; i++) // for BGR elements
				pixelColor[i] = gray;
			//ustawienie obliczonej wartości piksela na obrazie wyjściowym
			dst.at<Vec3b>(y, x) = pixelColor;
		}
}

//fukcja zmieniająca kontrast i jasność obrazu src i umieszczająca wynik na obrazie dst
void BrightnessAndContrast(Mat src, Mat dst, float A, int B)
{
	//pętla po wszystkich pikselach obrazu
	for (int x = 0; x < src.cols; x++)
		for (int y = 0; y < src.rows; y++)
		{
			Vec3b pixelColor = src.at<Vec3b>(y, x);
			for (int i = 0; i < 3; i++) // for BGR elements
				pixelColor[i] = 255 - pixelColor[i];

			dst.at<Vec3b>(y, x) = pixelColor;
		}
}
// wartość jasności (B)
int brightness_value = 100;
//wartość kontrastu (A)
int alpha_value = 200;			

// funkcja związana z suwakiem, wywoływana przy zmianie jego położenia
void BrightnessAndContrastCallBack(int pos, void* userdata)
{
	Mat* img = (Mat*)userdata;
	//wywołanie funkcji realizującej zmianę jasności i kontrastu BrightnessAndContrast
	BrightnessAndContrast(srcImage, *img, alpha_value / 100.0f, brightness_value - 200);
	imshow("Bright Image", *img);
}

void Binarization(Mat src, Mat& dst, uchar threshold)
{
	dst = Mat::zeros(src.size(), CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			Vec3b pixel = src.at<Vec3b>(y, x);
			uchar J = (uchar)(0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0]);
			dst.at<uchar>(y, x) = (J > threshold) ? 255 : 0;
		}
	}
}

void PseudoThresholding(Mat src, Mat& dst, uchar threshold)
{
	dst = Mat::zeros(src.size(), CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			Vec3b pixel = src.at<Vec3b>(y, x);
			uchar J = (uchar)(0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0]);
			dst.at<uchar>(y, x) = (J > threshold) ? J : 0;
		}
	}
}

void DoubleThresholding(Mat src, Mat& dst, uchar t1, uchar t2)
{
	dst = Mat::zeros(src.size(), CV_8UC1);
	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			Vec3b pixel = src.at<Vec3b>(y, x);
			uchar J = (uchar)(0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0]);
			dst.at<uchar>(y, x) = (J >= t1 && J <= t2) ? 255 : 0;
		}
	}
}

uchar ComputeGradientThreshold(Mat src)
{
	float sum_G = 0.0f;
	float sum_JG = 0.0f;

	for (int y = 1; y < src.rows - 1; y++)
	{
		for (int x = 1; x < src.cols - 1; x++)
		{
			Vec3b pixel = src.at<Vec3b>(y, x);
			uchar J = (uchar)(0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0]);

			Vec3b pixel_xp = src.at<Vec3b>(y, x + 1);
			Vec3b pixel_xm = src.at<Vec3b>(y, x - 1);
			Vec3b pixel_yp = src.at<Vec3b>(y + 1, x);
			Vec3b pixel_ym = src.at<Vec3b>(y - 1, x);

			uchar Jxp = (uchar)(0.299f * pixel_xp[2] + 0.587f * pixel_xp[1] + 0.114f * pixel_xp[0]);
			uchar Jxm = (uchar)(0.299f * pixel_xm[2] + 0.587f * pixel_xm[1] + 0.114f * pixel_xm[0]);
			uchar Jyp = (uchar)(0.299f * pixel_yp[2] + 0.587f * pixel_yp[1] + 0.114f * pixel_yp[0]);
			uchar Jym = (uchar)(0.299f * pixel_ym[2] + 0.587f * pixel_ym[1] + 0.114f * pixel_ym[0]);

			float Gx = Jxp - Jxm;
			float Gy = Jyp - Jym;

			float G = max(abs(Gx), abs(Gy));

			sum_G += G;
			sum_JG += J * G;
		}
	}

	uchar t = (sum_G > 0) ? (uchar)(sum_JG / sum_G) : 0;

	cout << "Wyliczony prog t = " << (int)t << endl;
	cout << "Suma gradientow sum_G = " << sum_G << endl;
	cout << "Suma iloczynow jasnosci i gradientow sum_JG = " << sum_JG << endl;

	return t;
}

void CountHistogram(Mat src, float p[256])
{
	// zerujemy histogram
	for (int i = 0; i < 256; i++)
		p[i] = 0.0f;

	int width = src.cols;
	int height = src.rows;

	// liczenie histogramu (jasność J)
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			Vec3b pixel = src.at<Vec3b>(y, x);
			uchar J = (uchar)(0.299f * pixel[2] + 0.587f * pixel[1] + 0.114f * pixel[0]);
			p[J] += 1.0f;
		}
	}

	// normalizacja histogramu
	float total = width * height;
	for (int i = 0; i < 256; i++)
		p[i] /= total;
}


uchar ComputeIterativeThreshold(Mat src)
{
	float p[256];
	CountHistogram(src, p);

	// szukanie Jmin i Jmax
	int Jmin = 0, Jmax = 255;
	for (int i = 0; i < 256; i++)
	{
		if (p[i] > 0.0f)
		{
			Jmin = i;
			break;
		}
	}
	for (int i = 255; i >= 0; i--)
	{
		if (p[i] > 0.0f)
		{
			Jmax = i;
			break;
		}
	}

	float t = (Jmin + Jmax) / 2.0f;
	float t_old;
	bool end = false;

	while (!end)
	{
		float mi_0 = 0.0f, P_0 = 0.0f;
		float mi_1 = 0.0f, P_1 = 0.0f;

		for (int i = 0; i < (int)t; i++) {
			mi_0 += i * p[i];
			P_0 += p[i];
		}

		for (int i = (int)t + 1; i < 256; i++) {
			mi_1 += i * p[i];
			P_1 += p[i];
		}

		if (P_0 > 0) mi_0 /= P_0;
		if (P_1 > 0) mi_1 /= P_1;

		t_old = t;
		t = (mi_0 + mi_1) / 2.0f;

		if (abs(t - t_old) < 2.0f)
			end = true;
	}

	cout << "Iteracyjny prog t = " << (int)t << " (Jmin = " << Jmin << ", Jmax = " << Jmax << ")" << endl;
	return (uchar)t;
}


int main()
{
	// wczytanie obrazu do srcImage
	int r = read_image("Samples/Kwiat.bmp", &srcImage);
	if (r == -1) return(-1);
	ShowImageAt("Source image", srcImage, 0, 0);

	// Binaryzacja obrazu: próg 20% z 255 = 51
	Mat binaryImage;
	uchar threshold = 51;
	Binarization(srcImage, binaryImage, threshold);
	ShowImageAt("Binary image", binaryImage, 900, 0);

	// Pseudoprogowanie
	Mat pseudoImage;
	PseudoThresholding(srcImage, pseudoImage, 100);
	ShowImageAt("Pseudoprogowanie", pseudoImage, 1300, 0);

	// Progowanie z dwoma progami
	Mat doubleThreshImage;
	DoubleThresholding(srcImage, doubleThreshImage, 50, 100);
	ShowImageAt("2-Progowanie", doubleThreshImage, 1000, 500);

	// Binaryzacja na podstawie gradientu
	uchar gradientThreshold = ComputeGradientThreshold(srcImage);
	Mat gradientBin;
	Binarization(srcImage, gradientBin, gradientThreshold);
	ShowImageAt("Gradient Threshold", gradientBin, 100, 500);

	// Binaryzacja iteracyjna
	uchar iterativeThreshold = ComputeIterativeThreshold(srcImage);
	Mat iterativeBin;
	Binarization(srcImage, iterativeBin, iterativeThreshold);
	ShowImageAt("Iteracyjna Binaryzacja", iterativeBin, 600, 500);

	waitKey();
}