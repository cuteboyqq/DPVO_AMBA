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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <sched.h>
#include <ctype.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <time.h>
#include <signal.h>
#include <stdint.h>
#include <eazyai.h>
#include <math.h>

#include <queue>
#include <mutex>
#include "config_reader.hpp"
#include "wnc_app.hpp"
#include "path_utils.hpp"
#include "video_handler.hpp"
// System includes
#include <filesystem>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>
namespace fs = std::filesystem;


// Ambarella
#define FILENAME_LENGTH				(256)
#define TIME_MEASURE_LOOPS			(10)
#define DEFAULT_DETECTION_PYD_IDX	(1)

// WNC
#define DATA_FOLDER_PATH  "./data"
#define CONFIG_FILE_PATH	"./config/app_config.txt"


enum {
	DRAW_MODE_VOUT,
	DRAW_MODE_STREAM,
	DRAW_MODE_FILE,
};

#define DEFAULT_DRAW_MODE		(DRAW_MODE_STREAM)

ea_display_line_params_t params; // cooper code

EA_LOG_DECLARE_LOCAL(EA_LOG_LEVEL_NOTICE);

typedef struct global_param_s {
	uint32_t channel_id;
	uint32_t stream_id;
	uint32_t log_level;
	uint32_t detection_pyd_idx;
	int draw_mode;

	char input_dir[FILENAME_LENGTH];
	char output_dir[FILENAME_LENGTH];

	float display_w;
	float display_h;
	ea_display_t* display;

	ea_img_resource_t* img_resource;
} global_param_t;

volatile int run_flag = 1;

#define NO_ARG 0
#define HAS_ARG 1
enum numeric_short_options {
	INPUT_DIR,
	OUTPUT_DIR,
};

static struct option long_options[] = {

	{"in_dir", 			HAS_ARG, 0, INPUT_DIR},

	{"channel_id", 	HAS_ARG, 0, 'i'},
	{"stream_id", 	HAS_ARG, 0, 's'},

	{"draw_mode", 	HAS_ARG, 0, 'm'},
	{"log_level", 	HAS_ARG, 0, 'd'},

	{"help", 				NO_ARG, 0, 'h'},
	{0, 0, 0, 0},
};

static const char *short_options = "i:s:m:d:h";

struct hint_s {
	const char *arg;
	const char *str;
};

static const struct hint_s hint[] = {
	// Baisc info of network
	{"", "\tThe path to folder containing JEPG images."},

	// Data source, destination
	{"", "\tSet live input channel id, [0~4]"},
	{"", "\tSet live overlay display stream id, [0~8]"},
	{"", "\tSet detection network source pyramid buffer layer id, [0~5]. Default 1."},

	// Draw OSD params
	{"", "\tSet result showing method, 0: VOUT, 1: Stream, 2: File, default is 1"},
	{"", "\tSet the log level."},
	{"", "\tPrint help info"},
};


// specific global variables
std::string inputDirPath;
std::deque<std::pair<ea_tensor_t*, int>> frameQueue;
std::mutex queueMutex;
std::condition_variable frameCondVar;
std::atomic<bool> terminateThreads(false);


//
void appThreadFunction(WNC_APP& wncApp);
void processInput(
	const std::string& inputPath, 
	std::shared_ptr<spdlog::logger> logger, 
	unsigned int& count, 
	std::mutex& frameProcessedMutex,
	std::condition_variable& frameProcessedCV, 
	bool& frameProcessed,
	global_param_t* G_param,
	WNC_APP* wncApp);
void processApp(
	const std::string& inputPath,
	const InputType& inputType,
	const std::string& configPath,
	const std::string& logFile, 
	global_param_t* G_param,
	std::shared_ptr<spdlog::logger> logger);	
