package kr.kro.prjectwwtp.controller;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import kr.kro.prjectwwtp.domain.TmsData;
import kr.kro.prjectwwtp.domain.responseDTO;
import kr.kro.prjectwwtp.persistence.DataRepository;
import lombok.RequiredArgsConstructor;

@RestController
@RequiredArgsConstructor
public class DataController {
	private final DataRepository dataRepo;
	
	@GetMapping("/api/data")
	public ResponseEntity<Object> getTest() {
		responseDTO res = responseDTO.builder()
				.bSuccess(true)
				.errorMsg(null)
				.build();
		List<TmsData> list = dataRepo.findAll();
		for(TmsData data : list)
			res.addData(data);
		return ResponseEntity.ok().body(res);
	}

}
