#include "logging.h"
#include "common/config.h"
#include "spdlog/sinks/null_sink.h"
#include "3rd_party/ExceptionWithCallStack.h"
#include <time.h>
#include <stdlib.h>
#ifdef __unix__
#include <signal.h>
#endif

#ifdef _MSC_VER
#define noinline __declspec(noinline)
#else
#define noinline __attribute__((noinline))
#endif

std::shared_ptr<spdlog::logger> createStderrLogger(
    const std::string& name,
    const std::string& pattern,
    const std::vector<std::string>& files,
    bool quiet) {
  std::vector<spdlog::sink_ptr> sinks;

  auto stderr_sink = spdlog::sinks::stderr_sink_mt::instance();

  if(!quiet)
    sinks.push_back(stderr_sink);

  for(auto&& file : files) {
    auto file_sink
        = std::make_shared<spdlog::sinks::simple_file_sink_st>(file, true);
    sinks.push_back(file_sink);
  }

  auto logger
      = std::make_shared<spdlog::logger>(name, begin(sinks), end(sinks));

  spdlog::register_logger(logger);
  logger->set_pattern(pattern);
  return logger;
}

bool setLoggingLevel(spdlog::logger& logger, std::string const level) {
  if(level == "trace")
    logger.set_level(spdlog::level::trace);
  else if(level == "debug")
    logger.set_level(spdlog::level::debug);
  else if(level == "info")
    logger.set_level(spdlog::level::info);
  else if(level == "warn")
    logger.set_level(spdlog::level::warn);
  else if(level == "err" || level == "error")
    logger.set_level(spdlog::level::err);
  else if(level == "critical")
    logger.set_level(spdlog::level::critical);
  else if(level == "off")
    logger.set_level(spdlog::level::off);
  else {
    logger.warn("Unknown log level '{}' for logger '{}'",
                level.c_str(),
                logger.name().c_str());
    return false;
  }
  return true;
}

static void setErrorHandlers();
void createLoggers(const marian::Config* options) {
  std::vector<std::string> generalLogs;
  std::vector<std::string> validLogs;

  if(options && options->has("log")) {
    generalLogs.push_back(options->get<std::string>("log"));
#ifndef _WIN32
    // can't open the same file twice in Windows for some reason
    validLogs.push_back(options->get<std::string>("log"));
#endif
  }

  if(options && options->has("valid-log")
     && !options->get<std::string>("valid-log").empty()) {
    validLogs.push_back(options->get<std::string>("valid-log"));
  }

  bool quiet = options && options->get<bool>("quiet");
  Logger general{
      createStderrLogger("general", "[%Y-%m-%d %T] %v", generalLogs, quiet)};
  Logger valid{
      createStderrLogger("valid", "[%Y-%m-%d %T] [valid] %v", validLogs, quiet)};

  if(options && options->has("log-level")) {
    std::string loglevel = options->get<std::string>("log-level");
    if(!setLoggingLevel(*general, loglevel))
      return;
    setLoggingLevel(*valid, loglevel);
  }

  if (options && options->has("log-time-zone")) {
    std::string timezone = options->get<std::string>("log-time-zone");
    if (timezone != "") {
#ifdef _WIN32
#define setenv(var, val, over) SetEnvironmentVariableA(var, val) // ignoring over flag
#endif
      setenv("TZ", timezone.c_str(), true);
      tzset();
    }
  }

  setErrorHandlers();
}

static void unhandledException() {
  if (std::current_exception()) {
    try {
      throw; // rethrow so that we can get access to what()
    }
    catch (const std::exception& e) {
      ABORT("Unhandled {}: {}", typeid(e).name(), e.what());
    }
    catch (...) {
      ABORT("Unhandled exception");
    }
  }
  else
    std::abort();
}

static void setErrorHandlers() {
  // call stack for unhandled exceptions
  std::set_terminate(unhandledException);
#ifdef __unix__
  // catch segfaults
  struct sigaction sa = { 0 };
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_SIGINFO;
  sa.sa_sigaction = [](int /*signal*/, siginfo_t*, void*) { ABORT("Segmentation fault"); };
  sigaction(SIGSEGV, &sa, NULL);
  sa.sa_sigaction = [](int /*signal*/, siginfo_t*, void*) { ABORT("Floating-point exception"); };
  sigaction(SIGFPE, &sa, NULL);
#endif
}

// modify the log pattern for the "general" logger to include the MPI rank
// This is called upon initializing MPI. It is needed to associated error messages to ranks.
void switchtoMultinodeLogging(std::string nodeIdStr) {
  Logger log = spdlog::get("general");
  if (log)
    log->set_pattern("[%Y-%m-%d %T " + nodeIdStr + "] %v");
}


namespace marian {
  std::string getCallStack(size_t skipLevels) {
    return ::Microsoft::MSR::CNTK::DebugUtil::GetCallStack(skipLevels + 2, /*makeFunctionNamesStandOut=*/true);
  }

  void noinline logCallStack(size_t skipLevels) {
    checkedLog("general", "critical", getCallStack(skipLevels));
  }
}
