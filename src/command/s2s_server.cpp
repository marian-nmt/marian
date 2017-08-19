#include "marian.h"
#include "translator/beam_search.h"
#include "translator/translator.h"

#include "3rd_party/simple-websocket-server/server_ws.hpp"

typedef SimpleWeb::SocketServer<SimpleWeb::WS> WsServer;

int main(int argc, char **argv) {
  using namespace marian;

  // initialize translation model task
  auto options = New<Config>(argc, argv, ConfigMode::translating);
  auto task = New<TranslateServiceMultiGPU<BeamSearch>>(options);

  // create web service server
  WsServer server;
  server.config.port = options->get<size_t>("port");
  auto &translate = server.endpoint["^/translate/?$"];

  translate.on_message = [&task](Ptr<WsServer::Connection> connection,
                                 Ptr<WsServer::Message> message) {
    auto message_str = message->string();

    auto message_short = message_str;
    boost::algorithm::trim_right(message_short);
    LOG(info)->info("Message received: " + message_short);

    auto send_stream = std::make_shared<WsServer::SendStream>();
    boost::timer::cpu_timer timer;
    for(auto &transl : task->run({message_str})) {
      *send_stream << transl << std::endl;
    }
    LOG(info)->info("Translation took: {}", timer.format(5, "%ws"));

    connection->send(send_stream, [](const SimpleWeb::error_code &ec) {
      if(ec) {
        auto ec_str = std::to_string(ec.value());
        LOG(warn)
            ->warn("Error sending message: (" + ec_str + ") " + ec.message());
      }
    });
  };

  // Error Codes for error code meanings
  // http://www.boost.org/doc/libs/1_55_0/doc/html/boost_asio/reference.html
  translate.on_error = [](Ptr<WsServer::Connection> connection,
                          const SimpleWeb::error_code &ec) {
    auto ec_str = std::to_string(ec.value());
    LOG(warn)->warn("Connection error: (" + ec_str + ") " + ec.message());
  };

  // start server
  std::thread server_thread([&server]() {
    LOG(info)->info("Server started");
    server.start();
  });

  server_thread.join();

  return 0;
}
