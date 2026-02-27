package kr.kro.prjectwwtp.service;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import io.netty.channel.ChannelOption;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.handler.timeout.WriteTimeoutHandler;
import jakarta.annotation.PostConstruct;
import kr.kro.prjectwwtp.domain.fastApiResponseDTO;
import lombok.RequiredArgsConstructor;
import reactor.netty.http.client.HttpClient;

@Service
@RequiredArgsConstructor
public class FastApiService {
	private LogService logService;
	private WebClient webClient;
	
	@Value("${spring.FastAPI.URI}")
	private String fastAPIURI;
	
	@PostConstruct
	void initWebClient() {
		HttpClient httpClient = HttpClient.create()
				.option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000)
				.responseTimeout(Duration.ofSeconds(20))
				.doOnConnected(conn -> conn.addHandlerLast(new ReadTimeoutHandler(20, TimeUnit.SECONDS))
											.addHandlerLast(new WriteTimeoutHandler(20, TimeUnit.SECONDS)));
		webClient = WebClient.builder()
				.baseUrl(fastAPIURI)
				.clientConnector(new ReactorClientHttpConnector(httpClient))
				.build();
	}
	
	public fastApiResponseDTO getPredict(String uri, Object obj) {
		try {
			fastApiResponseDTO response = webClient.post()
				.uri(uri)
				.contentType(MediaType.APPLICATION_JSON)
				.bodyValue(obj)
				.retrieve()
				.bodyToMono(fastApiResponseDTO.class)
				.block();
			return response;
		} catch (WebClientResponseException e) {
			System.out.println("HTTP Status : " + e.getStatusCode());
			System.out.println("Response Headers : " + e.getHeaders());
			System.out.println("Response Body : " + e.getResponseBodyAsString());
			//System.out.println("Request : " + pIn);
			logService.addErrorLog("FastApiService.java", "getPredict()", e.getMessage());
			throw new RuntimeException("API 호출 실패", e);
		}
	}

}
