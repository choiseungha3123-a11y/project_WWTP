package kr.kro.prjectwwtp.domain;

import java.util.Map;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;

import io.swagger.v3.oas.annotations.media.Schema;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@AllArgsConstructor
@NoArgsConstructor
@Builder
@JsonInclude(JsonInclude.Include.NON_NULL)
@Schema(description = "FastAPI 서버 응답")
public class fastApiResponseDTO {
	@JsonProperty("request_id")
	@Schema(description = "요청 고유값", example = "550e8400-e29b-41d4-a716-446655440000")
	private String requestId;
	
	@JsonProperty("ok")
	@Schema(description = "요청 성공 여부", example = "true")
	private boolean ok;
	
	@JsonProperty("output")
	@Schema(description = "예측 결과 데이터")
	private Map<String, Object> output;
	
	@JsonProperty("latency_ms")
	@Schema(description = "API 응답 소요시간(밀리초)", example = "731")
	private Integer latencyMs;
	
	@JsonProperty("error")
	@Schema(description = "오류 메시지", example = "null")
	private String error;
}

