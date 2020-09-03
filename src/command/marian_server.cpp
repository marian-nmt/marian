#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"
#include "common/timer.h"
#include "common/utils.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WSServer;

int main(int argc, char **argv) {
  using namespace marian;

  // Initialize translation task
  auto options = parseOptions(argc, argv, cli::mode::server, true);
  auto task = New<TranslateService<BeamSearch>>(options);
  auto quiet = options->get<bool>("quiet-translation");

  // Initialize web server
  WSServer server;
  server.config.port = (short)options->get<size_t>("port", 8080);

  auto &translate = server.endpoint["^/translate/?$"];

  translate.on_message = [&task, quiet](Ptr<WSServer::Connection> connection,
                                        Ptr<WSServer::InMessage> message) {
    // Get input text
    auto inputText = message->string();
    auto sendStream = std::make_shared<WSServer::OutMessage>();

    // Translate
    timer::Timer timer;
    auto outputText = task->run(inputText);
    *sendStream << outputText << std::endl;
    if(!quiet)
      LOG(info, "Translation took: {:.5f}s", timer.elapsed());

    // Send translation back
    connection->send(sendStream, [](const SimpleWeb::error_code &ec) {
      if(ec)
        LOG(error, "Error sending message: ({}) {}", ec.value(), ec.message());
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WSServer::Connection> /*connection*/,
                          const SimpleWeb::error_code &ec) {
    LOG(error, "Connection error: ({}) {}", ec.value(), ec.message());
  };

  // Start server thread
  std::thread serverThread([&server]() {
    server.start([](unsigned short port) {
      LOG(info, "Server is listening on port {}", port);
    });
  });

  serverThread.join();

  return 0;
}
