package kr.kro.prjectwwtp.domain;

import java.time.LocalDateTime;

import jakarta.persistence.Column;
import jakarta.persistence.Entity;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Id;
import jakarta.persistence.Temporal;
import jakarta.persistence.TemporalType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import lombok.ToString;

@Getter
@Setter
@ToString
@Builder
@AllArgsConstructor
@NoArgsConstructor
@Entity
public class TmsData {
	@Id
	@GeneratedValue(strategy = GenerationType.IDENTITY)
	@Column(name = "data_no") // DB 컬럼명은 그대로 유지
	private long dataNo; // 필드명을 CamelCase로 변경
	@Temporal(TemporalType.TIMESTAMP)
	@Column(updatable = false)
	LocalDateTime time;
	int stn;
	
	double wd1;
	double wd2;
	double wds;
	double wss;
	double wd10;
	double ws10;
	double ta;
	double re;
	double rn15m;
	double rn60m;
	double rn12h;
	double rnday;
	double hm;
	double pa;
	double ps;
	double td; 
}
