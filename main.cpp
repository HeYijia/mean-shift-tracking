#include "meanshift.h"
#include <iostream>

cv::Rect rect;
bool drawing_rect = false;
bool selected = false;
void mouse_cb(int event,int x,int y,int flag,void* param);

int main()
{
    cv::VideoCapture frame_capture;
    frame_capture = cv::VideoCapture("Homework_video.avi");

    cv::Mat frame,temp_img;
    frame_capture.read(frame);
    cv::namedWindow("image_show");
    temp_img = frame.clone();
    cv::setMouseCallback("image_show",mouse_cb,(void*) &temp_img);
    while(selected == false)
    {
        cv::Mat temp = temp_img.clone()  ;
        if( drawing_rect )
            cv::rectangle( temp, rect,cv::Scalar(0,0,255),2);
        cv::imshow("image_show", temp );

        if( cv::waitKey( 15 )==27 )
            break;
    }

    // creat meanshift obj
    MeanShift ms;
	// init the meanshift
    ms.Init_target_frame(frame,rect);

    cv::VideoWriter writer("tracking_result.avi",CV_FOURCC('M','J','P','G'),20,cv::Size(frame.cols,frame.rows));

	while(1)
	{

        if(!frame_capture.read(frame))
			break;	

        // tracking
        cv::Rect ms_rect =  ms.track(frame);

        // show the tracking reslut;
        cv::rectangle(frame,ms_rect,cv::Scalar(0,0,255),3);
        writer<< frame;
        cv::imshow("image_show",frame);
        cv::waitKey(25);

	}

    cv::waitKey( 0);
    cv::destroyWindow("image_show");
    return 0;
}

void mouse_cb(int event,int x,int y,int flag,void* param)
{
    cv::Mat *image = (cv::Mat*) param;
    switch( event ){
        case CV_EVENT_MOUSEMOVE:
            if( drawing_rect ){
                rect.width = x-rect.x;
                rect.height = y-rect.y;
            }
            break;

        case CV_EVENT_LBUTTONDOWN:
            drawing_rect = true;
            rect = cv::Rect( x, y, 0, 0 );
            break;

        case CV_EVENT_LBUTTONUP:
            drawing_rect = false;
            if( rect.width < 0 ){
                rect.x += rect.width;
                rect.width *= -1;
            }
            if( rect.height < 0 ){
                rect.y += rect.height;
                rect.height *= -1;
            }
            cv::rectangle(*image,rect,cv::Scalar(0),2);
            selected = true;
            break;
    }

}
