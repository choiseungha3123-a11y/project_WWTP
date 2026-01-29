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
	@Column(name = "time", updatable = false)
	private LocalDateTime time;
	@Column(name = "stn")
	private int stn;
	
	@Column(name = "wd1")
	private double wd1;
	@Column(name = "wd2")
	private double wd2;
	@Column(name = "wds")
	private double wds;
	@Column(name = "wss")
	private double wss;
	@Column(name = "wd10")
	private double wd10;
	@Column(name = "ws10")
	private double ws10;
	@Column(name = "ta")
	private double ta;
	@Column(name = "re")
	private double re;
	@Column(name = "rn15m")
	private double rn15m;
	@Column(name = "rn60m")
	private double rn60m;
	@Column(name = "rn12h")
	private double rn12h;
	@Column(name = "rnday")
	private double rnday;
	@Column(name = "hm")
	private double hm;
	@Column(name = "pa")
	private double pa;
	@Column(name = "ps")
	private double ps;
	@Column(name = "td")
	private double td; 
}
