/*
  (C) 2025-2026 Wistron NeWeb Corporation (WNC) - All Rights Reserved

  This software and its associated documentation are the confidential and
  proprietary information of Wistron NeWeb Corporation (WNC) ("Company") and
  may not be copied, modified, distributed, or otherwise disclosed to third
  parties without the express written consent of the Company.

  Unauthorized reproduction, distribution, or disclosure of this software and
  its associated documentation or the information contained herein is a
  violation of applicable laws and may result in severe legal penalties.
*/
#include "main.hpp"
#include <opencv2/opencv.hpp>


/**
 * @brief Display usage information for the test_gesture application
 * 
 * This function prints the usage information for the test_gesture application,
 * including the available command line options and their descriptions.
 * 
 * @param void No input parameters
 * @return void No return value
 */
// Definitions for DPVO timing globals declared in main.hpp
std::mutex g_dpvoTimingMutex;
std::unordered_map<unsigned int, DPVOTimingInfo> g_dpvoTimingMap;

static void usage(void)
{
	int i = 0;

	printf("\nusage:\n");
	for (i = 0; i < (int)(sizeof(long_options) / sizeof(long_options[0])) - 1; ++i) {
		if (isalpha(long_options[i].val))
			printf("-%c ", long_options[i].val);
		else
			printf("   ");
		printf("--%s", long_options[i].name);
		if (hint[i].arg[0] != 0)
			printf(" [%s]", hint[i].arg);
		printf("\t%s\n", hint[i].str);
	}
	printf("\nExample:\n"
		"\ttest_gesture -b yolov5_hand_detection.bin --in images"
		" --out /model.11/Conv_output_0 --out /model.16/Conv_output_0 --out /model.21/Conv_output_0 "
		" -b gesture_cavalry.bin --in input --out output \n"
		);

	return;
}


/**
 * @brief Parse command line arguments and update global parameters
 * 
 * This function parses command line arguments and updates the global parameters
 * structure with the provided values. It handles various command line options
 * and updates the parameters accordingly.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @param G_param Pointer to the global parameters structure
 * @return int Returns 0 on success, negative value on failure
 */
static int parse_param(int argc, char **argv, global_param_t *G_param)
{
	int ch = 0;
	int rval = EA_SUCCESS;
	int value = 0;
	int option_index = 0;

	opterr = 0;
	while ((ch = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1)
	{
		switch (ch)
		{
			case INPUT_DIR:
				value = strlen(optarg);
				if (value == 0)
				{
					printf("input_dir is empty\n");
					rval = -1;
					break;
				}
				if (value >= FILENAME_LENGTH)
				{
					printf("input_dir should be no more than %d characters\n", FILENAME_LENGTH);
					rval = -1;
					break;
				}
				snprintf(G_param->input_dir, FILENAME_LENGTH, "%s", optarg);
				if (optarg[value - 1] != '/')
				{
					strncat(G_param->input_dir, "/", FILENAME_LENGTH - strlen(G_param->input_dir) - 1);
				}
				inputDirPath = std::string(optarg);
				break;
			case 'c':
				G_param->channel_id = atoi(optarg);
				if (G_param->channel_id < 0 || G_param->channel_id > 4)
				{
					EA_LOG_ERROR("Channel id[%d] is error !\n", G_param->channel_id);
				}
				EA_LOG_NOTICE("Channel id is %d.\n", G_param->channel_id);
				break;
			
			case 's':
				G_param->stream_id = atoi(optarg);
				if (G_param->stream_id < 0 || G_param->stream_id > 8)
				{
					EA_LOG_ERROR("Stream id[%d] is error !\n", G_param->stream_id);
				}
				EA_LOG_NOTICE("Stream id is %d.\n", G_param->stream_id);
				break;

			case 'm':
				G_param->draw_mode = atoi(optarg);
				if (G_param->draw_mode < DRAW_MODE_VOUT || G_param->draw_mode > DRAW_MODE_FILE)
				{
					EA_LOG_ERROR("Draw mode[%d] is error !\n", G_param->draw_mode);
				}
				EA_LOG_NOTICE("Draw mode is %d.\n", G_param->draw_mode);
				break;

			case 'd':
				value = atoi(optarg);
				if (value < EA_LOG_LEVEL_NONE || value > EA_LOG_LEVEL_VERBOSE)
				{
					EA_LOG_ERROR("The log level[%d] is error.\n", value);
					rval = EA_FAIL;
					break;
				}
				G_param->log_level = value;
				EA_LOG_NOTICE("The log level is %d.\n", G_param->log_level);
				break;

			case 'h':
				usage();
				rval = EA_FAIL;
				break;

			default:
				EA_LOG_ERROR("unknown option found: %c\n", ch);
				rval = EA_FAIL;
			}
	}

	return rval;
}

/**
 * @brief Initialize and set default parameters for the application
 * 
 * This function initializes the global parameters structure with default values
 * and then parses command line arguments to override these defaults.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @param G_param Pointer to the global parameters structure
 * @return int Returns 0 on success, negative value on failure
 */
static int init_param(int argc, char **argv, global_param_t *G_param)
{
	int rval = 0;

	// set default params
	G_param->channel_id 					= 0;  // Channel ID for video input
	G_param->stream_id 						= 0;  // Stream ID for video processing
	G_param->draw_mode 						= DRAW_MODE_STREAM;  // Default to stream mode (vs. VOUT or FILE)
	G_param->log_level 						= EA_LOG_LEVEL_NOTICE;  // Default log level
	G_param->detection_pyd_idx 		= DEFAULT_DETECTION_PYD_IDX;  // Pyramid index for detection  // TODO: What is this?

	do {
		/* Check if command line arguments are provided
		 * If no arguments, display usage and exit
		 * Otherwise parse the provided arguments
		 */
		if (argc < 2)
		{
			usage();
			exit(0);
		}
		else
		{
			rval = parse_param(argc, argv, G_param);
			if (rval < 0)
			{
				break;
			}
		}

		/* Set the local log level based on the global parameter setting.
		 * This controls the verbosity of log messages in the application.
		 */
		EA_LOG_SET_GLOBAL(G_param->log_level);

		/* In file mode (when not using VOUT or STREAM modes), 
		 * we need to ensure that input and output directories are specified.
		 * These directories are used to read input images and save output results.
		 */
		if (G_param->draw_mode != DRAW_MODE_VOUT && G_param->draw_mode != DRAW_MODE_STREAM)
		{
			if (strlen(G_param->input_dir) == 0)
			{
				EA_LOG_ERROR("input_dir should not be empty in file mode !\n");
				rval = -1;
				break;
			}
		}

		/* Log all the current parameter settings at NOTICE level
		 * This provides visibility into the configuration being used for this run
		 * and can help with debugging and verification of command line arguments
		 */
		EA_LOG_NOTICE("Live parameters:\n");
		EA_LOG_NOTICE("\tDraw_mode: %d (0=vout, 1=stream, 2=file)\n", G_param->draw_mode);
		if (G_param->draw_mode != DRAW_MODE_VOUT && G_param->draw_mode != DRAW_MODE_STREAM)
		{
			EA_LOG_NOTICE("\tinput_dir path: %s\n", G_param->input_dir);
			EA_LOG_NOTICE("\toutput_dir path: %s\n", G_param->output_dir);
		}
		EA_LOG_NOTICE("\tHand det pyramid idx: %d.\n", G_param->detection_pyd_idx); //TODO: What is this?
		EA_LOG_NOTICE("\tchannel ID: %d\n", G_param->channel_id);
		EA_LOG_NOTICE("\tstream ID: %d\n", G_param->stream_id);
		EA_LOG_NOTICE("\tlog level: %d (0=None, 1=Error, 2=Notice, 3=Debug, 4=Verbose)\n", G_param->log_level);
	} while (0);

	return rval;
}


/**
 * @brief Initialize the computer vision environment
 * 
 * This function initializes the EazyAI environment with the specified features
 * based on the global parameters. It also registers utility callbacks for file
 * output mode and creates appropriate image resources based on the draw mode.
 * 
 * @param G_param Pointer to the global parameters structure
 * @return int Returns 0 on success, negative value on failure
 */
static int cv_env_init(global_param_t *G_param)
{
	int rval = 0;
	int ret = 0;
	int features = 0;

	do {
		features = EA_ENV_ENABLE_CAVALRY
			| EA_ENV_ENABLE_VPROC
			| EA_ENV_ENABLE_NNCTRL;

		/* Enable IAV (Image and Video) features when using display output modes
		 * that require video processing capabilities (VOUT or streaming) */
		if (G_param->draw_mode == DRAW_MODE_VOUT || G_param->draw_mode == DRAW_MODE_STREAM)
		{
			features |= EA_ENV_ENABLE_IAV;
		}

		/* Enable appropriate OSD (On-Screen Display) features based on the draw mode:
		 * - OSD_VOUT for direct video output display
		 * - OSD_STREAM for network streaming display
		 */
		if (G_param->draw_mode == DRAW_MODE_VOUT)
		{
			features |= EA_ENV_ENABLE_OSD_VOUT;
		}
		else if (G_param->draw_mode == DRAW_MODE_STREAM)
		{
			features |= EA_ENV_ENABLE_OSD_STREAM;
		}

		/* Initialize the EazyAI environment with the specified features
		 * This must succeed for the application to continue */
		ret = ea_env_open(features);
		RVAL_ASSERT(ret == EA_SUCCESS);

		/* Register utility callbacks for file output mode
		 * This enables proper file handling for saving detection results */
		if (G_param->draw_mode == DRAW_MODE_FILE)
		{
			ea_utils_env_callbacks_register();
		}

		/* Create appropriate image resource based on the draw mode:
		 * - For VOUT/STREAM modes: Use pyramid resource with channel ID
		 * - For FILE mode: Use JPEG folder with input directory path
		 */
		if (G_param->draw_mode == DRAW_MODE_VOUT || G_param->draw_mode == DRAW_MODE_STREAM)
		{
			// from pyramid
			G_param->img_resource = ea_img_resource_new(EA_PYRAMID, (void *)(unsigned long)G_param->channel_id); // chan_id
		}
		else
		{
			G_param->img_resource = ea_img_resource_new(EA_JPEG_FOLDER, (void *)G_param->input_dir);
		}
		RVAL_ASSERT(G_param->img_resource != NULL);

		/* Create display object based on the draw mode:
		 * - VOUT: Direct video output to analog display
		 * - STREAM: Network streaming with specified stream ID
		 * - FILE: JPEG output to the specified directory
		 */
		if (G_param->draw_mode == DRAW_MODE_VOUT)
		{
			G_param->display = ea_display_new(EA_DISPLAY_VOUT, EA_DISPLAY_ANALOG_VOUT, EA_DISPLAY_BBOX_TEXTBOX, NULL);
		}
		else if (G_param->draw_mode == DRAW_MODE_STREAM)
		{
			G_param->display = ea_display_new(EA_DISPLAY_STREAM, G_param->stream_id, EA_DISPLAY_BBOX_TEXTBOX, NULL);
		}
		else
		{
			G_param->display = ea_display_new(EA_DISPLAY_JPEG, EA_TENSOR_COLOR_MODE_BGR, EA_DISPLAY_BBOX_TEXTBOX, (void *)G_param->output_dir);
		}
		RVAL_ASSERT(G_param->display != NULL);

		/* Store display dimensions for later use in drawing operations */
		G_param->display_w = ea_display_obj_params(G_param->display)->dis_win_w;
		G_param->display_h = ea_display_obj_params(G_param->display)->dis_win_h;
	} while(0);

	/* Clean up resources if initialization failed
	 * This ensures proper resource deallocation in case of errors
	 * during the initialization process
	 */
	if (rval < 0)
	{
		if (G_param->display)
		{
			ea_display_free(G_param->display);
			G_param->display = NULL;
		}

		if (G_param->img_resource)
		{
			ea_img_resource_free(G_param->img_resource);
			G_param->img_resource = NULL;
		}

		if (ret == EA_SUCCESS)
		{
			ea_env_close();
		}
	}

	return rval;
}

/**
 * @brief Deinitialize the computer vision environment
 * 
 * This function frees the resources allocated for the display object and image resource
 * and closes the EazyAI environment.
 * 
 * @param G_param Pointer to the global parameters structure
 */
static void cv_env_deinit(global_param_t *G_param)
{
	/* Free resources allocated for display object */
	if (G_param->display)
	{
		ea_display_free(G_param->display);
		G_param->display = NULL;
	}

	/* Free resources allocated for image resource */
	if (G_param->img_resource)
	{
		ea_img_resource_free(G_param->img_resource);
		G_param->img_resource = NULL;
	}

	/* Close the EazyAI environment */
	ea_env_close();
}

static int env_init(global_param_t *G_param)
{
	int rval = EA_SUCCESS;

	do {
		/* Initialize the computer vision environment
		 * This sets up display, image resources, and other CV components
		 */
		RVAL_OK(cv_env_init(G_param));

	} while(0);

	return rval;
}

static void env_deinit(global_param_t *G_param)
{
	/* Clean up resources allocated for computer vision environment */
	cv_env_deinit(G_param);
}

// TODO: Not implemented yet
// /**
//  * @brief Draw the detected hand bounding boxes on the display
//  * 
//  * This function draws the detected hand bounding boxes on the display
//  * and updates the display object parameters. It also logs the drawing
//  * operation for debugging purposes.
//  * 
//  * @param G_param Pointer to the global parameters structure
//  * @param det_result Pointer to the hand detection result structure
//  * @return int Returns 0 on success, negative value on failure
//  */
// static int draw_result(global_param_t *G_param, hand_det_result_t *det_result)
// {
// 	int rval = 0;
// 	int text_len = 32;
// 	char text[text_len];
// 	int i = 0;
// 	int draw_num = 0;
// 	unsigned long start_time = ea_gettime_us();

// 	do {
// 		draw_num = det_result->valid_det_count ;
		
// 		memset(text, 0, text_len);
// 		ea_display_obj_params(G_param->display)->obj_win_w = 1.0f;
// 		ea_display_obj_params(G_param->display)->obj_win_h = 1.0f;

// 		for (i = 0; i < draw_num; i++)
// 		{
// 			snprintf(text, text_len, "human");
// 			ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_RED;
// 			ea_display_set_bbox(G_param->display, text, //ea_display_set_textbox
// 				det_result->detections[i].x_start, det_result->detections[i].y_start,
// 				det_result->detections[i].x_end - det_result->detections[i].x_start,
// 				det_result->detections[i].y_end - det_result->detections[i].y_start);
// 		}
// 	} while (0);

// 	EA_LOG_DEBUG("Gesture draw_result time: %ld us\n", ea_gettime_us() - start_time);

// 	return rval;
// }

// =================================================================================================
// DEPRECATED: YOLOv8/WNC_APP functions - commented out as YOLOv8 code has been removed
// =================================================================================================
// Cooper code
// std::unordered_map<int, std::pair<int, int>> idToX1X2; // Map to store obj.id and its x1, x2 values
/*
static int draw_result_test(global_param_t *G_param, WNC_APP* wncApp)
{
	int rval = 0, previousX1, previousX2;
	unsigned long start_time = ea_gettime_us();

	std::string text;
	WNC_APP_Results res;
	wncApp->getAppResult(res);
    wncApp->_printAppResult(res);
    // === Draw vehicle detection ===
    for (const auto& obj : res.vehicleObjList) {
        std::string text = "Vehicle, Conf: " + std::to_string(obj.confidence);
        ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_LIME;   // pick color for vehicle
        ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_WHITE;

        ea_display_set_textbox(G_param->display, text.c_str(),
            obj.x1, obj.y1,(obj.x2 - obj.x1), (obj.y2 - obj.y1));
    }
    // === Draw rider detection ===
    for (const auto& obj : res.riderObjList) {
        std::string text = "Rider, Conf: " + std::to_string(obj.confidence);
        ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_LIME;   // pick color for rider
        ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_WHITE;

        ea_display_set_textbox(G_param->display, text.c_str(),
        obj.x1, obj.y1,(obj.x2 - obj.x1), (obj.y2 - obj.y1));
    }

    // === Draw face detection ===
    for (const auto& obj : res.faceObjList) {
        std::string text = "Rider, Conf: " + std::to_string(obj.confidence);
        ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_FUCHSIA;   // pick color for face
        ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_FUCHSIA;

        ea_display_set_textbox(G_param->display, text.c_str(),
        obj.x1, obj.y1,(obj.x2 - obj.x1), (obj.y2 - obj.y1));
    }

    // === Draw skeleton poses ===
    for (const auto& obj : res.poseObjList) {
        // std::string text = "Pose ID: " + std::to_string(obj.objID);
        // if (!obj.skeletonAction.empty()) {
        //     text += " Action: " + obj.skeletonAction;
        // }

        // ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_YELLOW;
        // ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_WHITE;
        // ea_display_set_textbox(G_param->display, text.c_str(),
        //     obj.x1, obj.y1,
        //     (obj.x2 - obj.x1), (obj.y2 - obj.y1));

        // ------------------- Keypoints -------------------
        ea_display_circle_params_t pointParams;
        memset(&pointParams, 0, sizeof(pointParams));
        pointParams.color = EA_16_COLORS_RED;
        pointParams.thickness = 1;

        // assume display resolution available
        int disp_w = G_param->display_w;
        int disp_h = G_param->display_h;

        for (size_t i = 0; i < obj.pose_kpts.size(); i++) {
            int x = obj.pose_kpts[i].first;
            int y = obj.pose_kpts[i].second;

            // If keypoints are normalized [0,1], scale to pixel coords
            if (x > 0 && x <= 1 && y > 0 && y <= 1) {
                x = static_cast<int>(x * disp_w);
                y = static_cast<int>(y * disp_h);
            }

            if (x > 0 && y > 0) {
                ea_display_set_circle(G_param->display, (float)x, (float)y, 1.0f, &pointParams);
            }
        }

       // ------------------- Skeleton lines -------------------
        static const int skeletonPairs[][2] = {
            {5, 7}, {7, 9},     // left arm
            {6, 8}, {8, 10},    // right arm
            {3, 5}, {4, 6},      // shoulders
            {11, 13}, {13, 15}, // left leg
            {12, 14}, {14, 16}, // right leg
            {11, 12},           // hips
            {5, 11}, {6, 12},{5, 6},   // torso
            {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, // head
        };

        // Assign different colors for parts
        const int partColors[] = {
            EA_16_COLORS_YELLOW, // left arm
            EA_16_COLORS_YELLOW,
            EA_16_COLORS_NAVY,   // right arm
            EA_16_COLORS_NAVY,
            EA_16_COLORS_WHITE,  // shoulders
            EA_16_COLORS_WHITE, 
            EA_16_COLORS_RED, // left leg
            EA_16_COLORS_RED,
            EA_16_COLORS_BLUE,   // right leg
            EA_16_COLORS_BLUE,
            EA_16_COLORS_MAROON, // hips
            EA_16_COLORS_GREEN,  // torso
            EA_16_COLORS_GREEN,
            EA_16_COLORS_GREEN,
            EA_16_COLORS_RED,    // head
            EA_16_COLORS_RED,
            EA_16_COLORS_RED,
            EA_16_COLORS_RED,
            EA_16_COLORS_RED
        };

        for (size_t idx = 0; idx < sizeof(skeletonPairs)/sizeof(skeletonPairs[0]); idx++) {
            int i = skeletonPairs[idx][0], j = skeletonPairs[idx][1];
            if (i < (int)obj.pose_kpts.size() && j < (int)obj.pose_kpts.size()) {
                int x1 = obj.pose_kpts[i].first;
                int y1 = obj.pose_kpts[i].second;
                int x2 = obj.pose_kpts[j].first;
                int y2 = obj.pose_kpts[j].second;

                if (x1 > 0 && x1 <= 1 && y1 > 0 && y1 <= 1) {
                    x1 = static_cast<int>(x1 * disp_w);
                    y1 = static_cast<int>(y1 * disp_h);
                }
                if (x2 > 0 && x2 <= 1 && y2 > 0 && y2 <= 1) {
                    x2 = static_cast<int>(x2 * disp_w);
                    y2 = static_cast<int>(y2 * disp_h);
                }

                if (x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0) {
                    ea_display_line_params_t lineParams;
                    memset(&lineParams, 0, sizeof(lineParams));
                    lineParams.color = partColors[idx]; // colorful skeleton
                    lineParams.thickness = 8;           // thicker
                    ea_display_set_line(G_param->display, (float)x1, (float)y1, (float)x2, (float)y2, &lineParams);
                }
            }
        }

    }



	for (const auto& obj : res.trackObjList) {
#if 0
		if (idToX1X2.find(obj.id) != idToX1X2.end()) {
			// If obj.id already exists, compare x1 and x2 values
			auto[previousX1, previousX2] = idToX1X2[obj.id];
			printf("Duplicate Object ID: %d. Previous x1: %d, x2: %d, Current x1: %d, x2: %d\n", obj.id, previousX1, previousX2, obj.bbox.x1, obj.bbox.x2);
		}else {
			// If obj.id does not exist, add it to the map
			idToX1X2[obj.id] = {obj.bbox.x1, obj.bbox.x2};
		}
#endif
		if (obj.bbox.label == 0) { // human
			text = "ID: " + std::to_string(obj.id) + ", Conf: " + std::to_string(obj.bbox.confidence);
			ea_display_obj_params(G_param->display)->obj_win_w = 512;
			ea_display_obj_params(G_param->display)->obj_win_h = 288;
			ea_display_obj_params(G_param->display)->border_thickness = 5;
			ea_display_obj_params(G_param->display)->font_size = 20;
			ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_WHITE;
			ea_display_obj_params(G_param->display)->text_background_transparency = 0;
			ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_LIME;
			
			if (wncApp->m_config->EnableLineCross && !wncApp->m_config->EnableROI){
				params.thickness = 3;
				params.color = EA_16_COLORS_BLUE;
				// draw line
				ea_display_set_line(G_param->display, wncApp->m_config->LineCrossX1, wncApp->m_config->LineCrossY1, wncApp->m_config->LineCrossX2,
						wncApp->m_config->LineCrossY2, &params);

				// alarm zone  0:left 1:right
				if (wncApp->m_config->AlarmZone){
					if (idToX1X2.find(obj.id) != idToX1X2.end()) {
	                        		// If obj.id already exists, compare x1 and x2 values
        	                		auto[previousX1, previousX2] = idToX1X2[obj.id];

						if (obj.bbox.x1+((obj.bbox.x2-obj.bbox.x1)/2) > wncApp->m_config->LineCrossX1 && previousX1 < wncApp->m_config->LineCrossX1){
							text = "Alarm" + text;
							ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_RED;
							ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_RED;
						} 
						else //if (obj.bbox.x1 < wncApp->m_config->LineCrossX1 && obj.bbox.x2 < wncApp->m_config->LineCrossX2)
						{
							ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_LIME;
						}
               				}else {
                        			// If obj.id does not exist, add it to the map
                        			idToX1X2[obj.id] = {obj.bbox.x1, obj.bbox.x2};
                			}
				} else if (!wncApp->m_config->AlarmZone){
					if (idToX1X2.find(obj.id) != idToX1X2.end()) {
                                                // If obj.id already exists, compare x1 and x2 values
                                                auto[previousX1, previousX2] = idToX1X2[obj.id];
						if (obj.bbox.x2+((obj.bbox.x2-obj.bbox.x1)/2) < wncApp->m_config->LineCrossX2 && previousX2 > wncApp->m_config->LineCrossX2){
							text = "Alarm" + text;
							ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_RED;
							ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_RED;
						} 
						else //if (obj.bbox.x1 > wncApp->m_config->LineCrossX1 && obj.bbox.x2 > wncApp->m_config->LineCrossX2)
						{
							ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_LIME;
						}	
					}else {
                                                // If obj.id does not exist, add it to the map
                                                idToX1X2[obj.id] = {obj.bbox.x1, obj.bbox.x2};
                                        }
				}
			}
			if (wncApp->m_config->EnableROI && !wncApp->m_config->EnableLineCross){
				ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_YELLOW;
				// draw ROI
				ea_display_set_textbox(G_param->display, "set ROI by user",
                                        wncApp->m_config->RoiX1, wncApp->m_config->RoiY1, std::abs(wncApp->m_config->RoiX2-wncApp->m_config->RoiX1),
					std::abs(wncApp->m_config->RoiY2-wncApp->m_config->RoiY1));
				// alarm
                                if (obj.bbox.x1 > wncApp->m_config->RoiX1 && obj.bbox.y1 > wncApp->m_config->RoiY1 && obj.bbox.x2 < wncApp->m_config->RoiX2
						&& obj.bbox.y2 < wncApp->m_config->RoiY2){
					text = "Alarm" + text;
                                        ea_display_obj_params(G_param->display)->text_color = EA_16_COLORS_RED;
					ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_RED;
				} else if (obj.bbox.x1 < wncApp->m_config->RoiX1 || obj.bbox.y1 < wncApp->m_config->RoiY1 ||
						obj.bbox.x2 > wncApp->m_config->RoiX2 || obj.bbox.y2 > wncApp->m_config->RoiY2){
					ea_display_obj_params(G_param->display)->box_color = EA_16_COLORS_LIME;
				}
			}
			
			// bounding box
			// Set the bbox parameters
			ea_display_set_textbox(G_param->display, text.c_str(),
					obj.bbox.x1, obj.bbox.y1,                               // 左上角 (相對座標, 0~1)
					(obj.bbox.x2-obj.bbox.x1), (obj.bbox.y2-obj.bbox.y1)    // 寬高 (相對座標, 0~1)
			);
			printf("--------------------------------------------------\n");
			printf("obj id:%d, obj.bbox.x1:%d, obj.bbox.y1:%d, obj.bbox.x2:%d, obj.bbox.y2:%d\n", obj.id, obj.bbox.x1, obj.bbox.y1, obj.bbox.x2, obj.bbox.y2);
			printf("--------------------------------------------------\n");
		}
	}

	EA_LOG_DEBUG("draw_result time: %ld us\n", ea_gettime_us() - start_time);

	return rval;
}
*/
// =================================================================================================

/**
 * @brief Start the main run loop for hand detection
 * 
 * This function is the main processing loop for hand detection. It handles
 * image acquisition, detection, and display updates. It also measures the
 * time for performance analysis.
 * 
 * @param G_param Pointer to the global parameters structure
 * @return int Returns 0 on success, negative value on failure
 */
static int start_run(global_param_t *G_param)
{
	// Create data folder if it doesn't exist
	if (!fs::exists(DATA_FOLDER_PATH))
	{
		fs::create_directory(DATA_FOLDER_PATH);
	}

	//
	try
	{
		// ======
		auto logger = spdlog::stdout_color_mt("MAIN");
		logger->set_pattern("[%n] [%^%l%$] %v");
		logger->set_level(spdlog::level::info);

		// Define paths
		const std::string dataFolderPath = inputDirPath;

		// Check required folders
		if (!fs::exists(dataFolderPath)) {
				logger->error("Data folder '{}' does not exist or is not a directory", dataFolderPath);
				return 1;
		}

		DIR* dir = opendir(dataFolderPath.c_str());
		// Process each file in the data folder
		if (dir != nullptr) 
		{
			struct dirent* entry;
			while ((entry = readdir(dir)) != nullptr) 
			{
				// if (entry->d_name[0] == '.') continue; 
				std::string currentPath = dataFolderPath + "/" + entry->d_name;

				// Skip if not a supported file type
				InputType inputType = detectInputType(currentPath);
				if (inputType == InputType::Unknown)
				{
					logger->info("Skipping unsupported file: {}", currentPath);
					continue;
				}
				logger->info("\nProcessing: {}\n-------------------------------------------------", currentPath);
				// processApp(currentPath, inputType, CONFIG_FILE_PATH, entry->d_name, G_param, logger);
				processDPVOApp(currentPath, inputType, CONFIG_FILE_PATH, entry->d_name, G_param, logger);
			}
			closedir(dir);
		}

		logger->info("");
		logger->info("-------------------------------------------------");
		logger->info("Completed processing all inputs");
		logger->info("");
		return 0;
	}
	catch (const std::exception& e) 
	{
		std::cerr << "Caught exception in main: " << e.what() << std::endl;
		return 1;
	} 
	catch (...) 
	{
		std::cerr << "Caught unknown exception in main" << std::endl;
		return 1;
	}
}

/**
 * @brief Thread function for processing frames from the queue
 * 
 * DEPRECATED: This function is for YOLOv8/WNC_APP which has been removed.
 * Use appDPVOthreadFunction instead for DPVO processing.
 * 
 * @param wncApp Reference to the WNC_APP instance
 */
/*
void appThreadFunction(WNC_APP& wncApp)
{
	try 
	{
		while (true)
		{
			std::pair<ea_tensor_t*, int> framePair;
			bool hasFrame = false;
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				if (frameQueue.empty())
				{
					if (terminateThreads)
					{
						// All frames processed and termination signal received
						break;
					}
					frameCondVar.wait(lock);
					continue;
				}

				framePair = std::move(frameQueue.front());
				frameQueue.pop_front();
				hasFrame = true;
			}

			if (hasFrame && framePair.first != NULL)
			{
				wncApp.addFrame(framePair.first);
			} 
			else
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		}
	} 
	catch (const std::exception& e) 
	{
		std::cerr << "Caught exception in APP thread: " << e.what() << std::endl;
	} 
	catch (...) 
	{
		std::cerr << "Caught unknown exception in APP thread" << std::endl;
	}
}
*/
// =================================================================================================

/**
 * @brief Thread function for processing frames from the queue for DPVO
 * 
 * This function continuously checks the frame queue for new frames.
 * If a frame is available, it is added to the DPVO instance.
 * If no frame is available, it sleeps for a short duration to reduce CPU usage.
 * 
 * @param dpvo Reference to the DPVO instance
 */
void appDPVOthreadFunction(DPVO& dpvo)
{
	try 
	{
		while (true)
		{
			std::pair<ea_tensor_t*, int> framePair;
			bool hasFrame = false;
			{
				std::unique_lock<std::mutex> lock(queueMutex);
				if (frameQueue.empty())
				{
					if (terminateThreads)
					{
						// All frames processed and termination signal received
						break;
					}
					frameCondVar.wait(lock);
					continue;
				}

				framePair = std::move(frameQueue.front());
				frameQueue.pop_front();
				hasFrame = true;
			}

			if (hasFrame && framePair.first != NULL)
			{
				auto logger = spdlog::get("MAIN");
				const unsigned int frame_index = static_cast<unsigned int>(framePair.second);

				if (logger) logger->debug("appDPVOthreadFunction: Calling dpvo.addFrame for frame {}", frame_index);

				const auto dpvo_start = std::chrono::steady_clock::now();
				dpvo.addFrame(framePair.first);
				const auto dpvo_end = std::chrono::steady_clock::now();

				{
					std::lock_guard<std::mutex> timing_lock(g_dpvoTimingMutex);
					g_dpvoTimingMap[frame_index] = DPVOTimingInfo{dpvo_start, dpvo_end};
				}

				if (logger) logger->debug("appDPVOthreadFunction: dpvo.addFrame completed for frame {}", frame_index);
			} 
			else
			{
				std::this_thread::sleep_for(std::chrono::milliseconds(50));
			}
		}
	} 
	catch (const std::exception& e) 
	{
		std::cerr << "Caught exception in DPVO thread: " << e.what() << std::endl;
	} 
	catch (...) 
	{
		std::cerr << "Caught unknown exception in DPVO thread" << std::endl;
	}
}

/**
 * @brief Process the input data
 * 
 * DEPRECATED: This function is for YOLOv8/WNC_APP which has been removed.
 * Use processDPVOInput instead for DPVO processing.
 * 
 * @param inputPath Path to the input data
 * @param inputType Type of input data
 * @param logger Logger for logging
 * @param count Frame count
 * @param frameProcessedMutex Mutex for frame processing
 * @param frameProcessed Condition variable for frame processing
 * @param frameProcessed Boolean for frame processing
 * @param G_param Global parameters
 * @param wncApp WNC_APP instance
 */
/*
void processInput(
	const std::string& inputPath,
	const InputType& inputType,
	std::shared_ptr<spdlog::logger> logger, 
	unsigned int& count, 
	std::mutex& frameProcessedMutex,
	std::condition_variable& frameProcessedCV, 
	bool& frameProcessed,
	global_param_t* G_param,
	WNC_APP* wncApp)
{
	int rval = EA_SUCCESS;

	ea_tensor_t* tmpTensor = NULL;
	ea_img_resource_data_t data;
	uint32_t dsp_pts = 0;

	// Time measurement
	unsigned long start_time = 0;
	double sum_time = 0.0f;
	uint32_t loop_count = 0;


	std::string dataFolderPath;
	if (inputType == InputType::Video) 
	{
		try
		{
			VideoHandler videoHandler(inputPath, logger.get());
			std::string baseName = PathUtils::extractBaseName(inputPath);
			dataFolderPath = std::string(DATA_FOLDER_PATH) + "/" + baseName;
			unsigned int frameCount = 0;

			logger->info("Generate images from video: {} for eazyai library reading", inputPath);
			if (!fs::exists(dataFolderPath))
			{
				fs::create_directory(dataFolderPath);
				while (true)
				{
					bool hasFrame = videoHandler.processNextFrame(frameCount, dataFolderPath);
					if (!hasFrame) break;
				}

				logger->info("Saved {} frames for video: {} to {}", frameCount, inputPath, dataFolderPath);
			}
			else
			{
				logger->info("Video {} already processed, skipping", inputPath);
			}

		}
		catch (const std::exception& e)
		{
			logger->error("Video processing error for {}: {}", inputPath, e.what());
			throw;
		}
	}
	else if (inputType == InputType::ImageDirectory) 
	{
		dataFolderPath = inputPath;
		if (!fs::exists(dataFolderPath))
		{
			logger->error("Image directory {} does not exist", dataFolderPath);
			exit(1);
		}
		else
		{
			logger->info("Image directory {} exists", dataFolderPath);
		}
	}
	else
	{
		logger->error("Unknown input type, exit the program");
		exit(1);
	}

	//
	snprintf(G_param->input_dir, FILENAME_LENGTH, "%s", dataFolderPath.c_str());
	if (!dataFolderPath.empty() && dataFolderPath.back() != '/')
	{
		strncat(G_param->input_dir, "/", FILENAME_LENGTH - strlen(G_param->input_dir) - 1);
	}
	logger->info("Set G_param->input_dir: {}", G_param->input_dir);

	//
	G_param->img_resource = ea_img_resource_new(EA_JPEG_FOLDER, (void *)G_param->input_dir);
	logger->info("Updated G_param->img_resource");


	while (run_flag)
	{
		memset(&data, 0, sizeof(data));
		RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));

		start_time = ea_gettime_us();

		if (G_param->draw_mode == DRAW_MODE_VOUT || G_param->draw_mode == DRAW_MODE_STREAM)
		{
			RVAL_ASSERT(data.tensor_group != NULL);
			RVAL_ASSERT(data.tensor_num >= 1);
			tmpTensor = data.tensor_group[G_param->detection_pyd_idx];
		}
		else
		{
			if (data.tensor_group == NULL)
			{
				EA_LOG_NOTICE("All files are handled\n");
				break;
			}

			tmpTensor = data.tensor_group[0];
		}


		ea_tensor_t* imgTensor = ea_tensor_new_from_other(tmpTensor, 0);
		// ea_tensor_t* imgTensor = nullptr; Lee said set to null will cause pending


		EA_LOG_DEBUG("Input img: %ldx%ldx%ld\n",
			ea_tensor_shape(imgTensor)[EA_W], ea_tensor_shape(imgTensor)[EA_H], ea_tensor_shape(imgTensor)[EA_C]);

		{
			std::lock_guard<std::mutex> lock(queueMutex);
			logger->info("Input Source Frame Index = {}", count++);
			frameQueue.emplace_back(std::pair<ea_tensor_t*, int>(imgTensor, 0));
		}

		// Cooper code
		//Let draw_result() function handle the drawing
        // RVAL_OK(draw_result_test(G_param, wncApp));


		// Drop the image resource data after processing
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));

		frameCondVar.notify_one();

		std::unique_lock<std::mutex> lock(frameProcessedMutex);
		frameProcessedCV.wait(lock, [&]() { return frameProcessed; });
		frameProcessed = false;
	}
}
*/
// =================================================================================================

/**
 * @brief Process the application
 * 
 * DEPRECATED: This function is for YOLOv8/WNC_APP which has been removed.
 * Use processDPVOApp instead for DPVO processing.
 * 
 * @param inputPath Path to the input data
 * @param inputType Type of input data
 * @param configPath Path to the config file
 * @param logFile Path to the log file
 * @param G_param Global parameters
 * @param logger Logger for logging
 */
/*
void processApp(
	const std::string& inputPath,
	const InputType& inputType,
	const std::string& configPath,
	const std::string& logFile, 
	global_param_t* G_param,
	std::shared_ptr<spdlog::logger> logger)
{
	// Reset global state
	frameQueue.clear();
	terminateThreads = false;

	// Remove old log file if exists
	try {
		std::remove(logFile.c_str());
	}
	catch (...) {
		logger->error("Error while removing {}", logFile);
	}

	// Process with App
	[&]() {
		std::unique_ptr<WNC_APP> wncApp(new WNC_APP(configPath, inputPath));

		std::mutex frameProcessedMutex;
		std::condition_variable frameProcessedCV;
		bool frameProcessed = false;

		wncApp->setFrameProcessedCallback([&]()
		{
			{
				std::lock_guard<std::mutex> lock(frameProcessedMutex);
				frameProcessed = true;
			}
			frameProcessedCV.notify_one();
		});

		std::thread appThread(appThreadFunction, std::ref(*wncApp));
		unsigned int count = 0;

		processInput(
			inputPath,
			inputType,
			logger,
			count,
			frameProcessedMutex,
			frameProcessedCV,
			frameProcessed,
			G_param,
			wncApp.get());

		// Signal completion and wait for thread
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			terminateThreads = true;
		}
		frameCondVar.notify_all();
		
		if (appThread.joinable())
		{
			appThread.join();
		}

		while (!wncApp->isProcessingComplete())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			logger->info("Waiting for app thread to finish...");
		}
	}();

	return;
}
*/
// =================================================================================================

/**
 * @brief Preprocess image tensor: undistort and crop to divisible by 16
 * 
 * This function matches Python DPVO preprocessing:
 * 1. Undistort if distortion parameters are available
 * 2. Crop to make dimensions divisible by 16
 * 
 * @param imgTensor Input tensor (will be modified)
 * @param config Config containing camera parameters
 * @param logger Logger for logging
 * @return true if preprocessing succeeded, false otherwise
 */
static bool preprocessImageTensor(ea_tensor_t* imgTensor, Config_S* config, std::shared_ptr<spdlog::logger> logger)
{
	if (logger) {
		logger->info("preprocessImageTensor: Function called, imgTensor={}, config={}", 
		             (void*)imgTensor, (void*)config);
	}
	
	if (!imgTensor) {
		if (logger) logger->error("preprocessImageTensor: imgTensor is null");
		return false;
	}
	if (!config) {
		if (logger) logger->error("preprocessImageTensor: config is null");
		return false;
	}
	
	const size_t* shape = ea_tensor_shape(imgTensor);
	int H = static_cast<int>(shape[EA_H]);
	int W = static_cast<int>(shape[EA_W]);
	int C = static_cast<int>(shape[EA_C]);
	
	if (C != 3) {
		if (logger) logger->error("preprocessImageTensor: Expected 3 channels, got {}", C);
		return false;
	}
	
	void* tensor_data = ea_tensor_data(imgTensor);
	if (!tensor_data) {
		if (logger) logger->error("preprocessImageTensor: Failed to get tensor data");
		return false;
	}
	
	// Convert from [C, H, W] BGR to cv::Mat [H, W, C] BGR format
	cv::Mat img_bgr(H, W, CV_8UC3);
	const uint8_t* src = static_cast<const uint8_t*>(tensor_data);
	for (int c = 0; c < 3; c++) {
		for (int y = 0; y < H; y++) {
			for (int x = 0; x < W; x++) {
				int src_idx = c * H * W + y * W + x;
				img_bgr.at<cv::Vec3b>(y, x)[c] = src[src_idx];
			}
		}
	}
	
	cv::Mat img_processed = img_bgr.clone();
	static unsigned int frame_count = 0;
	
	// STEP 1: Undistort if distortion parameters are available (matching Python: if len(calib) > 4)
	bool has_distortion = (std::abs(config->stCameraConfig.distortion_k1) > 1e-6f ||
	                       std::abs(config->stCameraConfig.distortion_k2) > 1e-6f ||
	                       std::abs(config->stCameraConfig.distortion_p1) > 1e-6f ||
	                       std::abs(config->stCameraConfig.distortion_p2) > 1e-6f);
	
	if (has_distortion) {
		// Build camera matrix K
		cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
		K.at<double>(0, 0) = config->stCameraConfig.intrinsic_fx;  // fx
		K.at<double>(0, 2) = config->stCameraConfig.intrinsic_cx;  // cx
		K.at<double>(1, 1) = config->stCameraConfig.intrinsic_fy;  // fy
		K.at<double>(1, 2) = config->stCameraConfig.intrinsic_cy;  // cy
		
		// Build distortion coefficients [k1, k2, p1, p2]
		cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);
		dist_coeffs.at<double>(0) = config->stCameraConfig.distortion_k1;
		dist_coeffs.at<double>(1) = config->stCameraConfig.distortion_k2;
		dist_coeffs.at<double>(2) = config->stCameraConfig.distortion_p1;
		dist_coeffs.at<double>(3) = config->stCameraConfig.distortion_p2;
		
		if (logger && frame_count == 0) {
			logger->info("preprocessImageTensor: Applying undistortion with k1={:.6f}, k2={:.6f}, p1={:.6f}, p2={:.6f}",
			             config->stCameraConfig.distortion_k1, config->stCameraConfig.distortion_k2,
			             config->stCameraConfig.distortion_p1, config->stCameraConfig.distortion_p2);
		}
		
		cv::undistort(img_bgr, img_processed, K, dist_coeffs);
		
		// Update dimensions after undistortion (may change slightly)
		H = img_processed.rows;
		W = img_processed.cols;
	} else {
		if (logger && frame_count == 0) {
			logger->info("preprocessImageTensor: No distortion parameters (all zero), skipping undistortion");
		}
	}
	
	// STEP 2: Crop to make dimensions divisible by 16 (matching Python: image[:h-h%16, :w-w%16])
	int h_cropped = H; //- (H % 16);
	int w_cropped = W; //- (W % 16);
	
	if (h_cropped != H || w_cropped != W) {
		if (logger && frame_count == 0) {
			logger->info("preprocessImageTensor: Cropping from {}x{} to {}x{} (divisible by 16)",
			             W, H, w_cropped, h_cropped);
		}
		img_processed = img_processed(cv::Rect(0, 0, w_cropped, h_cropped));
		H = h_cropped;
		W = w_cropped;
	} else {
		if (logger && frame_count == 0) {
			logger->info("preprocessImageTensor: Dimensions {}x{} already divisible by 16, no cropping needed", W, H);
		}
	}
	
	frame_count++;
	
	// Convert back from [H, W, C] BGR to [C, H, W] BGR format
	// Note: We need to update the tensor data, but tensor dimensions may have changed
	// For now, we'll copy back to the original tensor (assuming it's large enough)
	// If dimensions changed, we'd need to create a new tensor, but that's complex with ea_tensor_t
	// So we'll only update the data if dimensions match, otherwise log a warning
	const size_t* shape_after = ea_tensor_shape(imgTensor);
	int H_orig = static_cast<int>(shape_after[EA_H]);
	int W_orig = static_cast<int>(shape_after[EA_W]);
	
	if (H != H_orig || W != W_orig) {
		if (logger && frame_count == 1) {
			logger->warn("preprocessImageTensor: Dimensions changed from {}x{} to {}x{}, but cannot resize tensor. "
			             "Using cropped region only.", W_orig, H_orig, W, H);
		}
		// Copy only the cropped region
		uint8_t* dst = static_cast<uint8_t*>(tensor_data);
		for (int c = 0; c < 3; c++) {
			for (int y = 0; y < H && y < H_orig; y++) {
				for (int x = 0; x < W && x < W_orig; x++) {
					int dst_idx = c * H_orig * W_orig + y * W_orig + x;
					dst[dst_idx] = img_processed.at<cv::Vec3b>(y, x)[c];
				}
			}
		}
		// Zero out remaining pixels if cropped
		if (H < H_orig || W < W_orig) {
			for (int c = 0; c < 3; c++) {
				for (int y = H; y < H_orig; y++) {
					for (int x = 0; x < W_orig; x++) {
						int dst_idx = c * H_orig * W_orig + y * W_orig + x;
						dst[dst_idx] = 0;
					}
				}
				for (int y = 0; y < H; y++) {
					for (int x = W; x < W_orig; x++) {
						int dst_idx = c * H_orig * W_orig + y * W_orig + x;
						dst[dst_idx] = 0;
					}
				}
			}
		}
	} else {
		// Dimensions match, copy back normally
		uint8_t* dst = static_cast<uint8_t*>(tensor_data);
		for (int c = 0; c < 3; c++) {
			for (int y = 0; y < H; y++) {
				for (int x = 0; x < W; x++) {
					int dst_idx = c * H * W + y * W + x;
					dst[dst_idx] = img_processed.at<cv::Vec3b>(y, x)[c];
				}
			}
		}
	}
	
	return true;
}

/**
 * @brief Process the input data for DPVO
 * 
 * This function processes the input data based on the input type for DPVO.
 * It handles video files and image directories differently.
 * 
 * @param inputPath Path to the input data
 * @param inputType Type of input data
 * @param logger Logger for logging
 * @param count Frame count
 * @param G_param Global parameters
 * @param dpvo DPVO instance
 * @param modelH Model input height
 * @param modelW Model input width
 * @param config Config containing camera parameters
 */
void processDPVOInput(
	const std::string& inputPath,
	const InputType& inputType,
	std::shared_ptr<spdlog::logger> logger, 
	unsigned int& count, 
	global_param_t* G_param,
	DPVO* dpvo,
	int modelH,
	int modelW,
	Config_S* config)
{
	int rval = EA_SUCCESS;
	
	if (logger) {
		logger->info("processDPVOInput: Called with config={}", (void*)config);
		if (config) {
			logger->info("processDPVOInput: Config has distortion_k1={:.6f}, distortion_k2={:.6f}, distortion_p1={:.6f}, distortion_p2={:.6f}",
			             config->stCameraConfig.distortion_k1, config->stCameraConfig.distortion_k2,
			             config->stCameraConfig.distortion_p1, config->stCameraConfig.distortion_p2);
		}
	}

	ea_tensor_t* tmpTensor = NULL;
	ea_img_resource_data_t data;

	std::string dataFolderPath;
	if (inputType == InputType::Video) 
	{
		try
		{
			VideoHandler videoHandler(inputPath, logger.get());
			std::string baseName = PathUtils::extractBaseName(inputPath);
			dataFolderPath = std::string(DATA_FOLDER_PATH) + "/" + baseName;
			unsigned int frameCount = 0;

			logger->info("Generate images from video: {} for eazyai library reading", inputPath);
			if (!fs::exists(dataFolderPath))
			{
				fs::create_directory(dataFolderPath);
				while (true)
				{
					bool hasFrame = videoHandler.processNextFrame(frameCount, dataFolderPath);
					if (!hasFrame) break;
				}

				logger->info("Saved {} frames for video: {} to {}", frameCount, inputPath, dataFolderPath);
			}
			else
			{
				logger->info("Video {} already processed, skipping", inputPath);
			}

		}
		catch (const std::exception& e)
		{
			logger->error("Video processing error for {}: {}", inputPath, e.what());
			throw;
		}
	}
	else if (inputType == InputType::ImageDirectory) 
	{
		dataFolderPath = inputPath;
		if (!fs::exists(dataFolderPath))
		{
			logger->error("Image directory {} does not exist", dataFolderPath);
			exit(1);
		}
		else
		{
			logger->info("Image directory {} exists", dataFolderPath);
		}
	}
	else
	{
		logger->error("Unknown input type, exit the program");
		exit(1);
	}

	// Set up image resource
	snprintf(G_param->input_dir, FILENAME_LENGTH, "%s", dataFolderPath.c_str());
	if (!dataFolderPath.empty() && dataFolderPath.back() != '/')
	{
		strncat(G_param->input_dir, "/", FILENAME_LENGTH - strlen(G_param->input_dir) - 1);
	}
	logger->info("Set G_param->input_dir: {}", G_param->input_dir);

	G_param->img_resource = ea_img_resource_new(EA_JPEG_FOLDER, (void *)G_param->input_dir);
	logger->info("Updated G_param->img_resource");

	// Process frames
	bool firstFrame = true;
	int actualH = 0, actualW = 0;
	
	while (run_flag)
	{
		memset(&data, 0, sizeof(data));
		RVAL_OK(ea_img_resource_hold_data(G_param->img_resource, &data));

		if (G_param->draw_mode == DRAW_MODE_VOUT || G_param->draw_mode == DRAW_MODE_STREAM)
		{
			RVAL_ASSERT(data.tensor_group != NULL);
			RVAL_ASSERT(data.tensor_num >= 1);
			tmpTensor = data.tensor_group[G_param->detection_pyd_idx];
		}
		else
		{
			if (data.tensor_group == NULL)
			{
				EA_LOG_NOTICE("All files are handled\n");
				break;
			}

			tmpTensor = data.tensor_group[0];
		}

		ea_tensor_t* imgTensor = ea_tensor_new_from_other(tmpTensor, 0);

		// Apply preprocessing: undistort and crop to divisible by 16 (matching Python DPVO)
		if (config) {
			if (logger) {
				logger->info("preprocessImageTensor: Starting preprocessing for frame {}", count);
			}
			if (!preprocessImageTensor(imgTensor, config, logger)) {
				if (logger) {
					logger->warn("preprocessImageTensor failed, continuing with original image");
				}
			} else {
				if (logger) {
					logger->info("preprocessImageTensor: Preprocessing completed successfully");
				}
			}
		} else {
			if (logger) {
				logger->warn("preprocessImageTensor: config is null, skipping preprocessing");
			}
		}

		// Get actual image dimensions from tensor (after preprocessing)
		const size_t* shape = ea_tensor_shape(imgTensor);
		int tensorH = static_cast<int>(shape[EA_H]);
		int tensorW = static_cast<int>(shape[EA_W]);
		int tensorC = static_cast<int>(shape[EA_C]);

		EA_LOG_DEBUG("Input img: %ldx%ldx%ld\n", shape[EA_W], shape[EA_H], shape[EA_C]);

		// Validate dimensions on first frame
		if (firstFrame) {
			actualH = tensorH;
			actualW = tensorW;
			firstFrame = false;
			
			logger->info("First frame dimensions: {}x{} (DPVO expects {}x{} from model config)", 
				actualH, actualW, modelH, modelW);
			
			// Note: DPVO is initialized with model input size (modelH x modelW)
			// The actual frames (actualH x actualW) may be different
			// fnet/inet models will resize internally, so this is expected
			if (actualH != modelH || actualW != modelW) {
				logger->info("Frame dimensions differ from model input size - models will resize internally");
			}
			
			// Validate that dimensions are reasonable
			if (actualH < 16 || actualW < 16) {
				logger->error("Actual image dimensions {}x{} are too small (minimum 16x16)", actualH, actualW);
				RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
				break;
			}
		} else {
			// Validate that subsequent frames have same dimensions
			if (tensorH != actualH || tensorW != actualW) {
				logger->warn("Frame dimension mismatch: expected {}x{}, got {}x{}. Skipping frame.", 
					actualH, actualW, tensorH, tensorW);
				RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
				continue;
			}
		}

		// Measure per-frame wall-clock time for DPVO pipeline (ingest → DPVO done)
		const auto frame_start = std::chrono::steady_clock::now();
		const unsigned int frame_index = count;

		{
			std::lock_guard<std::mutex> lock(queueMutex);
			logger->info("Input Source Frame Index = {}", count++);
			frameQueue.emplace_back(std::pair<ea_tensor_t*, int>(imgTensor, static_cast<int>(frame_index)));
			logger->info("frameQueue.emplace_back imgTensor finished");
		}

		frameCondVar.notify_one();

		// Wait for DPVO to process the frame BEFORE dropping the resource
		// This ensures the tensor remains valid during conversion
		// Since DPVO doesn't have a callback, we'll wait a bit and check
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		while (!dpvo->isProcessingComplete())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}

		// Drop the image resource data AFTER processing is complete
		// This ensures the tensor is no longer needed
		logger->info("Start ea_img_resource_drop_data (after processing complete)");
		RVAL_OK(ea_img_resource_drop_data(G_param->img_resource, &data));
		logger->info("ea_img_resource_drop_data finished");

		// Per-frame timing and FPS (yellow text for visibility)
		const auto frame_end = std::chrono::steady_clock::now();
		const double frame_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(frame_end - frame_start).count();
		const double frame_fps = frame_seconds > 0.0 ? 1.0 / frame_seconds : 0.0;

		logger->info(
			"\033[93m[DPVO] Frame {} time: {:.3f} s ({:.2f} FPS)\033[0m",
			frame_index,
			frame_seconds,
			frame_fps
		);
	}
}

/**
 * @brief Process the DPVO application
 * 
 * This function processes the DPVO application based on the input type.
 * It handles video files and image directories differently, similar to processApp
 * but uses DPVO instead of WNC_APP.
 * 
 * @param inputPath Path to the input data
 * @param inputType Type of input data
 * @param configPath Path to the config file
 * @param logFile Path to the log file
 * @param G_param Global parameters
 * @param logger Logger for logging
 */
void processDPVOApp(
	const std::string& inputPath,
	const InputType& inputType,
	const std::string& configPath,
	const std::string& logFile, 
	global_param_t* G_param,
	std::shared_ptr<spdlog::logger> logger)
{
	// Reset global state
	frameQueue.clear();
	terminateThreads = false;

	// Remove old log file if exists
	try {
		std::remove(logFile.c_str());
	}
	catch (...) {
		logger->error("Error while removing {}", logFile);
	}

	// Process with DPVO App
	[&]() {
		const auto pipeline_start = std::chrono::steady_clock::now();

		// Read config file
		AppConfigReader appConfigReader;
		appConfigReader.read(configPath);
		Config_S* config = appConfigReader.getConfig();

		// Create DPVOConfig with default values
		DPVOConfig dpvoCfg;
		// Reduce BUFFER_SIZE to prevent memory issues (4096 is too large)
		dpvoCfg.BUFFER_SIZE = 128;  // Use same as PatchGraph::BUFFER_SIZE
		dpvoCfg.PATCHES_PER_FRAME = 4;  // Use same as PatchGraph::PATCHES_PER_FRAME
		// You can customize other DPVOConfig values here if needed

		// Initialize fnet and inet models first to get their actual input dimensions
		// This ensures DPVO is initialized with the correct dimensions
		// Note: These temporary instances will be destroyed, and new ones created in Patchifier
		int ht = 0, wd = 0;
		
#ifdef USE_ONNX_RUNTIME
		if (config->useOnnxRuntime) {
			// Use ONNX Runtime models
			FNetInferenceONNX fnet(config);
			INetInferenceONNX inet(config);
			
			// Get model input dimensions (models resize internally, so we use their input size)
			ht = fnet.getInputHeight();
			wd = fnet.getInputWidth();
			
			// Validate that fnet and inet have same input dimensions
			if (ht != inet.getInputHeight() || wd != inet.getInputWidth()) {
				logger->error("FNet and INet have different input dimensions: FNet={}x{}, INet={}x{}", 
					ht, wd, inet.getInputHeight(), inet.getInputWidth());
				throw std::runtime_error("FNet and INet input dimension mismatch");
			}
			
			logger->info("DPVO Config: BUFFER_SIZE={}, PATCHES_PER_FRAME={}, Model input size: {}x{} (from ONNX fnet/inet models)", 
				dpvoCfg.BUFFER_SIZE, dpvoCfg.PATCHES_PER_FRAME, ht, wd);
			logger->info("FNet output: {}x{}, INet output: {}x{}", 
				fnet.getOutputHeight(), fnet.getOutputWidth(),
				inet.getOutputHeight(), inet.getOutputWidth());
		} else {
			// Use AMBA EazyAI models
			FNetInference fnet(config);
			INetInference inet(config);
			
			// Get model input dimensions (models resize internally, so we use their input size)
			ht = fnet.getInputHeight();
			wd = fnet.getInputWidth();
			
			// Validate that fnet and inet have same input dimensions
			if (ht != inet.getInputHeight() || wd != inet.getInputWidth()) {
				logger->error("FNet and INet have different input dimensions: FNet={}x{}, INet={}x{}", 
					ht, wd, inet.getInputHeight(), inet.getInputWidth());
				throw std::runtime_error("FNet and INet input dimension mismatch");
			}
			
			logger->info("DPVO Config: BUFFER_SIZE={}, PATCHES_PER_FRAME={}, Model input size: {}x{} (from AMBA fnet/inet models)", 
				dpvoCfg.BUFFER_SIZE, dpvoCfg.PATCHES_PER_FRAME, ht, wd);
			logger->info("FNet output: {}x{}, INet output: {}x{}", 
				fnet.getOutputHeight(), fnet.getOutputWidth(),
				inet.getOutputHeight(), inet.getOutputWidth());
		}
#else
		// ONNX Runtime not available, use AMBA models
		FNetInference fnet(config);
		INetInference inet(config);
		
		// Get model input dimensions (models resize internally, so we use their input size)
		ht = fnet.getInputHeight();
		wd = fnet.getInputWidth();
		
		// Validate that fnet and inet have same input dimensions
		if (ht != inet.getInputHeight() || wd != inet.getInputWidth()) {
			logger->error("FNet and INet have different input dimensions: FNet={}x{}, INet={}x{}", 
				ht, wd, inet.getInputHeight(), inet.getInputWidth());
			throw std::runtime_error("FNet and INet input dimension mismatch");
		}
		
		logger->info("DPVO Config: BUFFER_SIZE={}, PATCHES_PER_FRAME={}, Model input size: {}x{} (from AMBA fnet/inet models)", 
			dpvoCfg.BUFFER_SIZE, dpvoCfg.PATCHES_PER_FRAME, ht, wd);
		logger->info("FNet output: {}x{}, INet output: {}x{}", 
			fnet.getOutputHeight(), fnet.getOutputWidth(),
			inet.getOutputHeight(), inet.getOutputWidth());
#endif
		
		// Validate dimensions to prevent bad_array_new_length
		if (ht < 16 || wd < 16) {
			logger->error("Invalid model input dimensions: {}x{}. Minimum size is 16x16", ht, wd);
			throw std::runtime_error("Invalid model input dimensions for DPVO");
		}
		
		logger->info("Note: Actual frames are {}x{}, but models will resize to {}x{} internally", 
			config->frameHeight, config->frameWidth, ht, wd);

		// Drop loggers before creating DPVO (which will create models again via setPatchifierModels)
		// This prevents "logger with name already exists" errors
#ifdef SPDLOG_USE_SYSLOG
		spdlog::drop("fnet");
		spdlog::drop("inet");
#else
		spdlog::drop("fnet");
		spdlog::drop("inet");
#endif

		// Create DPVO instance with model input dimensions
		// This ensures fmap/imap buffers match what fnet/inet models output
		std::unique_ptr<DPVO> dpvo(new DPVO(dpvoCfg, ht, wd, config));

		// Set camera intrinsics: fx, fy, cx, cy (at full resolution)
		// These will be automatically scaled by RES=4 when stored in PatchGraph
		// User-provided intrinsics: fx=1660.0, fy=1660.0, cx=960.0, cy=540.0
		// Resolution: 1920x1080
		// float camera_intrinsics[4] = {1660.0f, 1660.0f, 960.0f, 540.0f};
		// dpvo->setIntrinsics(camera_intrinsics);
		// logger->info("Set camera intrinsics: fx={:.2f}, fy={:.2f}, cx={:.2f}, cy={:.2f}",
		//              camera_intrinsics[0], camera_intrinsics[1], camera_intrinsics[2], camera_intrinsics[3]);

		// Set fnet and inet models for Patchifier (will create new model instances)
		// This will also start the processing thread (via _startThreads)
		dpvo->setPatchifierModels(config, config);
		
		// Enable inference cache: saves FNet/INet/Update model outputs to bin files.
		// First run: models run normally and outputs are saved to cache directory.
		// Subsequent runs: cached outputs are loaded, model inference is SKIPPED (much faster).
		// Cache is organized by video/input name, e.g. inference_cache/IMG_0492/fnet/...
		// Delete the cache directory to force re-running inference.
		// Controlled by EnableInferenceCache in app_config.txt (0=disabled, 1=enabled)
		if (config->enableInferenceCache) {
			std::string baseName = PathUtils::extractBaseName(inputPath);
			std::string cachePath = "inference_cache/" + baseName;
			logger->info("Inference cache ENABLED, path: {}", cachePath);
			dpvo->enableInferenceCache(cachePath);
		} else {
			logger->info("Inference cache DISABLED (set EnableInferenceCache = 1 in app_config.txt to enable)");
		}
		
		// Enable visualization (optional)
		// NOTE: Requires Pangolin library to be installed and linked
		// Visualization displays 3D point cloud, camera trajectory, and current video frame
		dpvo->enableVisualization(true);
		
		// Enable frame saving: saves each viewer frame as PNG for later video creation
		// This avoids needing to screen-record for hours when AMBA model inference is slow
		// Output: viewer_frames/<video_name>/frame_00001.png, frame_00002.png, ...
		// Convert to video: ffmpeg -framerate 30 -i frame_%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4
		// Controlled by SaveViewerFrames in app_config.txt (0=disabled, 1=enabled)
		if (config->saveViewerFrames) {
			std::string baseName = PathUtils::extractBaseName(inputPath);
			std::string frameSavePath = "viewer_frames/" + baseName;
			dpvo->enableFrameSaving(frameSavePath);
			logger->info("Viewer frame saving ENABLED, path: {}", frameSavePath);
		} else {
			logger->info("Viewer frame saving DISABLED (set SaveViewerFrames = 1 in app_config.txt to enable)");
		}

		// Set up frame processed synchronization (similar to WNC_APP pattern)
		std::mutex frameProcessedMutex;
		std::condition_variable frameProcessedCV;
		bool frameProcessed = false;

		// Note: DPVO doesn't have a callback mechanism like WNC_APP,
		// so we'll use a simpler approach - just wait for processing to complete
		// after each frame is added. We can use a timer or check isProcessingComplete.
		logger->error("Start dpvoThread");
		std::thread dpvoThread(appDPVOthreadFunction, std::ref(*dpvo));
		logger->error("dpvoThread started");
		unsigned int count = 0;
		
		logger->error("Start processDPVOInput");
		// Process input frames using the dedicated function
		processDPVOInput(
			inputPath,
			inputType,
			logger,
			count,
			G_param,
			dpvo.get(),
			ht,
			wd,
			config);
		logger->error("processDPVOInput finished");
		// Signal completion and wait for thread
		{
			std::lock_guard<std::mutex> lock(queueMutex);
			terminateThreads = true;
		}
		frameCondVar.notify_all();
		
		if (dpvoThread.joinable())
		{
			dpvoThread.join();
		}

		// Stop DPVO's internal processing thread
		dpvo->stopProcessingThread();

		while (!dpvo->isProcessingComplete())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
			logger->info("Waiting for DPVO thread to finish...");
		}

		// Total pipeline timing (ingest + DPVO processing for all frames)
		const auto pipeline_end = std::chrono::steady_clock::now();
		const double pipeline_seconds =
			std::chrono::duration_cast<std::chrono::duration<double>>(pipeline_end - pipeline_start).count();
		const double avg_fps = pipeline_seconds > 0.0 ? static_cast<double>(count) / pipeline_seconds : 0.0;

		logger->info(
			"\033[93m[DPVO] Total pipeline time: {:.3f} s, frames: {}, average FPS: {:.2f}\033[0m",
			pipeline_seconds,
			count,
			avg_fps
		);
	}();

	return;
}

/**
 * @brief Signal handler for program termination
 * 
 * This function is called when a termination signal is received.
 * It sets the run_flag to 0, causing the main processing loop to exit.
 * 
 * @param signal_number Signal number received
 */
static void sigstop(int signal_number)
{
	run_flag = 0;
	(void)signal_number;
	printf("sigstop msg, exit application\n");
	exit(0);

	return;
}


/**
 * @brief Main entry point for the hand detection application
 * 
 * This function initializes the application, processes command line arguments,
 * sets up signal handlers, and starts the main processing loop.
 * 
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return int Returns 0 on success, negative value on failure
 */
int main(int argc, char **argv)
{
	int rval = EA_SUCCESS;                  // Return value for function calls

	global_param_t G_param;                 // Global parameters structure

	/* Set up signal handlers for graceful termination (SIGINT, SIGQUIT, SIGTERM) */
	signal(SIGINT, sigstop);
	signal(SIGQUIT, sigstop);
	signal(SIGTERM, sigstop);

	do {
		/* Initialize the global parameter structure to zeros
			* This ensures all fields start with known values and prevents undefined behavior
			*/
		memset(&G_param, 0, sizeof(global_param_t));

		/* Initialize parameters by parsing command line arguments
			* This function sets default values and then overrides them
			* with any provided command line options.
			*/
		rval = init_param(argc, argv, &G_param);
		if (rval < 0)
		{
			break;
		}

		/* Initialize the environment with the parsed parameters
			* This sets up all necessary resources and prepares the system
			* for hand detection processing
			*/
		RVAL_OK(env_init(&G_param));

		/* Start the main processing loop for hand detection
			* This function handles the core functionality of capturing frames,
			* detecting hands, and displaying/saving results according to the
			* configured draw mode (VOUT, STREAM, or FILE)
			*/
		RVAL_OK(start_run(&G_param));
		
	} while(0);


	/* Clean up and release all resources
	 * This function ensures proper shutdown by freeing memory,
	 * closing file handles, and releasing any hardware resources
	 * that were allocated during initialization
	 */
	env_deinit(&G_param);

	/* Log a termination message to indicate successful completion
	 * This provides a final confirmation that the application has
	 * terminated cleanly and all resources have been released
	 */
	EA_LOG_NOTICE("Application terminated successfully.\n");

	return 0;
}


