package kr.kro.prjectwwtp.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import kr.kro.prjectwwtp.domain.responseDTO;
import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
public class WeatherController {
	
	@GetMapping("/api/test")
	public ResponseEntity<Object> getTest(@RequestParam String message) {
		responseDTO res = responseDTO.builder()
				.bSuccess(true)
				.errorMsg(null)
				.build();
		res.addData(message);
		return ResponseEntity.ok().body(res);
	}

}
