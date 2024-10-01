#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

// Function to calculate Gaussian distribution
double calculateGaussianDistribution(double x, double sigma)
{
    // Calculate the Gaussian distribution value for a given x and standard deviation sigma
    return (1 / (sigma * sqrt(2 * 3.14159))) * exp(-(x * x) / (2 * sigma * sigma));
}

// Function to apply 1D Gaussian convolution
cv::Mat executeGaussianConvolution(cv::Mat input, double sigma, int kernelSize)
{
    // Create a kernel matrix with the given kernel size, single column, and double precision
    cv::Mat kernel(kernelSize, 1, CV_64F);
    double sum = 0; // Variable to accumulate the sum of kernel values for normalization

    // Calculate the Gaussian kernel values
    for (int i = 0; i < kernelSize; i++)
    {
        // Fill the kernel with Gaussian distribution values using a centered index
        kernel.at<double>(i, 0) = calculateGaussianDistribution(i - kernelSize / 2, sigma);
        sum += kernel.at<double>(i, 0); // Accumulate the sum for normalization
    }

    // Normalize the kernel by dividing each element by the sum of the kernel values
    kernel /= sum;

    // Create an output matrix with the same number of rows as the input and 1 column, using double precision
    cv::Mat output(input.rows, 1, CV_64F);
    int padding = kernelSize / 2; // Calculate padding for handling border cases during convolution

    // Perform convolution on the input matrix
    for (int i = 0; i < input.rows; i++)
    {
        double sum = 0; // Reset the sum for each output element
        for (int j = 0; j < kernelSize; j++)
        {
            int index = i + j - padding; // Calculate the corresponding index in the input matrix
            // Check if the index is within bounds of the input matrix
            if (index >= 0 && index < input.rows)
            {
                // Multiply the input value by the kernel value and accumulate the result
                sum += input.at<double>(index, 0) * kernel.at<double>(j, 0);
            }
        }
        // Store the convolved value in the output matrix
        output.at<double>(i, 0) = sum;
    }

    // Return the convolved output matrix
    return output;
}

int main()
{
    // Create a matrix of 6 rows and 1 column, filled with double precision values
    cv::Mat data(6, 1, CV_64F);
    // Initialize the values in the matrix to create a simple step function
    data.at<double>(0, 0) = 0.0;
    data.at<double>(1, 0) = 0.0;
    data.at<double>(2, 0) = 0.0;
    data.at<double>(3, 0) = 1.0;
    data.at<double>(4, 0) = 1.0;
    data.at<double>(5, 0) = 1.0;

    // Define sigma (standard deviation) for the Gaussian function
    double sigma = 1.0;
    // Define the kernel size for the Gaussian convolution
    int kernelSize = 3;

    // Apply the Gaussian convolution on the data matrix
    cv::Mat smoothedData = executeGaussianConvolution(data, sigma, kernelSize);

    // Output the original and smoothed data to the console
    std::cout << "Initial Data:" << std::endl;
    std::cout << data << std::endl; // Print the original data
    std::cout << "Smoothed Data:" << std::endl;
    std::cout << smoothedData << std::endl; // Print the smoothed data

    // Define the size of the image to visualize the data (400x400 pixels)
    int imageSize = 400;
    // Create a blank white image with 3 channels (color image)
    cv::Mat image(imageSize, imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    // Set the color for the original data (blue) using BGR format
    cv::Scalar initialColor(255, 0, 0);
    // Calculate the scaling factor for the x-axis (horizontal)
    int xScale = imageSize / data.rows;
    // Calculate the scaling factor for the y-axis (vertical)
    int yScale = imageSize / 2;

    // Draw the original data as a blue line on the image
    for (int i = 0; i < data.rows - 1; i++)
    {
        // Draw a line between consecutive points in the original data
        cv::line(image,
            cv::Point(i * xScale, imageSize - data.at<double>(i, 0) * yScale),
            cv::Point((i + 1) * xScale, imageSize - data.at<double>(i + 1, 0) * yScale),
            initialColor, 2); // Set line thickness to 2 pixels
    }

    // Set the color for the smoothed data (red) using BGR format
    cv::Scalar smoothedColor(0, 0, 255);
    // Draw the smoothed data as a red line on the image
    for (int i = 0; i < smoothedData.rows - 1; i++)
    {
        // Draw a line between consecutive points in the smoothed data
        cv::line(image,
            cv::Point(i * xScale, imageSize - smoothedData.at<double>(i, 0) * yScale),
            cv::Point((i + 1) * xScale, imageSize - smoothedData.at<double>(i + 1, 0) * yScale),
            smoothedColor, 2); // Set line thickness to 2 pixels
    }

    // Add text to the image to indicate the legend for the blue line (initial data)
    cv::putText(image, "Initial Data (Blue)", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, initialColor, 2);
    // Add text to the image to indicate the legend for the red line (smoothed data)
    cv::putText(image, "Smoothed Data (Red)", cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, smoothedColor, 2);

    // Display the final image in a window titled "Line Graph"
    cv::imshow("Line Graph", image);
    // Wait for the user to press a key to close the window
    cv::waitKey(0);
    // Close all OpenCV windows after the user presses a key
    cv::destroyAllWindows();

    // Wait for user input before exiting the program
    std::cin.get();
}