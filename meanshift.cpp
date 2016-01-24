/*
 * Based on paper "Kernel-Based Object Tracking"
 * you can find all the formula in the paper
*/

#include"meanshift.h"

MeanShift::MeanShift()
{
    cfg.MaxIter = 10;
    cfg.num_bins = 16;
    cfg.piexl_range = 256;
    _bin_width = cfg.piexl_range / cfg.num_bins;
}

bool  MeanShift::Init_target_frame(const cv::Mat &frame,const cv::Rect &rect)
{
    _target_Region = rect;
    _target_model = pdf_representation(frame,_target_Region);
    std::cout << "target model is build"<<std::endl;
}

float  MeanShift::Epanechnikov_kernel(cv::Mat &kernel)
{
    // epanechnikov function :   formula ( 12 ) in the paper
    //                   |-  param * (1 - x)  , x<1
    //      k(x) = |
    //                   |-  0                             otherwise

    int h = kernel.rows;
    int w = kernel.cols;

    float epanechnikov_cd = 0.1*PI*h*w;
    float kernel_sum = 0.0;
    for(int i=0;i<h;i++)
    {
        for(int j=0;j<w;j++)
        {
            float x = static_cast<float>(i - h/2);
            float  y = static_cast<float> (j - w/2);
            float norm_x = x*x/(h*h/4)+y*y/(w*w/4);
            float result =norm_x<1?(epanechnikov_cd*(1.0-norm_x)):0;
            kernel.at<float>(i,j) = result;
            kernel_sum += result;
        }
    }
    return kernel_sum;
}
cv::Mat MeanShift::pdf_representation(const cv::Mat &frame, const cv::Rect &rect)
{
    /* m-bin histograms be used*/

    // step 1: compute the  kernel k( ||x||^2 )
    cv::Mat kernel(rect.height,rect.width,CV_32F,cv::Scalar(0));
    float normalized_C = 1.0 / Epanechnikov_kernel(kernel);

    // step 2: compute p(y) = {pu},u=1,...,m
    //                                                pu = normalized_C * sum( k(||x||^2)*delta() ); delta is kronecker data function
    cv::Mat pdf_model(3,16,CV_32F,cv::Scalar(1e-10));

    cv::Vec3f curr_pixel_value;
    cv::Vec3f bin_value;

    int row_index = rect.y;
    int clo_index = rect.x;

    for(int i=0;i<rect.height;i++)
    {
        clo_index = rect.x;
        for(int j=0;j<rect.width;j++)
        {
            curr_pixel_value = frame.at<cv::Vec3b>(row_index,clo_index);
            bin_value[0] = (curr_pixel_value[0]/_bin_width);
            bin_value[1] = (curr_pixel_value[1]/_bin_width);
            bin_value[2] = (curr_pixel_value[2]/_bin_width);
            pdf_model.at<float>(0,bin_value[0]) += kernel.at<float>(i,j)*normalized_C;
            pdf_model.at<float>(1,bin_value[1]) += kernel.at<float>(i,j)*normalized_C;
            pdf_model.at<float>(2,bin_value[2]) += kernel.at<float>(i,j)*normalized_C;
            clo_index++;
        }
        row_index++;
    }

    return pdf_model;

}
cv::Mat MeanShift::CalWeight(const cv::Mat &frame, cv::Mat &target_model, cv::Mat &target_candidate, cv::Rect &rec)
{
    int rows = rec.height;
    int cols = rec.width;

    cv::Mat weight(rows,cols,CV_32F,cv::Scalar(1.0000));
    std::vector<cv::Mat> bgr_planes;
    split(frame, bgr_planes);
    int row_index = rec.y;
    int col_index = rec.x;

    // accroding to the paper:  wi = sum( sqrt( qu / pu) * delta( ))
    for(int k = 0; k < 3;  k++)
    {
        row_index = rec.y;
        for(int i=0;i<rows;i++)
        {
            col_index = rec.x;
            for(int j=0;j<cols;j++)
            {
                int curr_pixel = (bgr_planes[k].at<uchar>(row_index,col_index));
                int bin_value = curr_pixel/_bin_width;
                weight.at<float>(i,j) *= static_cast<float>((sqrt(target_model.at<float>(k, bin_value)/target_candidate.at<float>(k, bin_value))));
                col_index++;
            }
        row_index++;
        }
    }

    return weight;
}

cv::Rect MeanShift::track(const cv::Mat &next_frame)
{
    cv::Rect next_rect;
    for(int iter=0;iter<cfg.MaxIter;iter++)
    {
        // step 1: Derive the weights wi according to  formula (10) in the paper
        cv::Mat target_candidate = pdf_representation(next_frame,_target_Region);
        cv::Mat weight = CalWeight(next_frame,_target_model,target_candidate,_target_Region);

        float delta_x = 0.0;
        float sum_wij = 0.0;
        float delta_y = 0.0;
        float centre = static_cast<float>((weight.rows-1)/2.0);
        double mult = 0.0;

        // step 2 : find the next location of the target candidate according to (13):  a simple weighted average
        next_rect.x = _target_Region.x;
        next_rect.y = _target_Region.y;
        next_rect.width = _target_Region.width;
        next_rect.height = _target_Region.height;

        for(int i=0;i<weight.rows;i++)
        {
            for(int j=0;j<weight.cols;j++)
            {
                float norm_i = static_cast<float>(i-centre)/centre;
                float norm_j = static_cast<float>(j-centre)/centre;
                mult = pow(norm_i,2)+pow(norm_j,2)>1.0?0.0:1.0;
                //if(pow(norm_i,2)+pow(norm_j,2)>1)std::cout<<mult<<std::endl;
                delta_x += static_cast<float>(norm_j*weight.at<float>(i,j)*mult);
                delta_y += static_cast<float>(norm_i*weight.at<float>(i,j)*mult);
                sum_wij += static_cast<float>(weight.at<float>(i,j)*mult);
            }
        }

        next_rect.x += static_cast<int>((delta_x/sum_wij)*centre);
        next_rect.y += static_cast<int>((delta_y/sum_wij)*centre);

        if(abs(next_rect.x-_target_Region.x)<1 && abs(next_rect.y-_target_Region.y)<1)
        {
            break;
        }
        else
        {
            _target_Region.x = next_rect.x;
            _target_Region.y = next_rect.y;
        }
    }

    return next_rect;
}
